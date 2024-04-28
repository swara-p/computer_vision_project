import argparse
import cv2
import numpy as np
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import scipy.signal
import scipy.linalg
import scipy.sparse
import pyamg


class Blend:
    def __init__(self):
        pass

    def get_mask(self):
        pass

    def blend(self):
        pass


class CutPaste(Blend):
    def __init__(self, params):
        super().__init__()
        pass

    def get_mask(self,img):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        mask[:, :int(img.shape[1]/2)] = np.linspace(0, 1, int(img.shape[1]/2))
        mask = mask.astype(np.float32)

    def blend(self, src, tgt, mask=None):
        self.mask = self.get_mask() if mask is None else mask
        src, tgt = src.astype(float), tgt.astype(float)
        mask = cv2.blur(self.mask,(1,1))
        mask = mask.astype(np.float32) / 255.0
        res = mask*src + (1-mask)*tgt
        res = res.astype(np.uint8)
        return res
    
class AlphaBlend(Blend):
    def __init__(self, params):
        super().__init__()
        self.kernel = params['kernel']

    def get_mask(self,img):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        mask[:, :int(img.shape[1]/2)] = np.linspace(0, 1, int(img.shape[1]/2))
        mask = mask.astype(np.float32)

    def blend(self, src, tgt, mask=None):
        self.mask = self.get_mask() if mask is None else mask
        src, tgt = src.astype(float), tgt.astype(float)
        mask = cv2.blur(self.mask, self.kernel)
        mask = mask.astype(np.float32) / 255.0
        res = mask * src + (1 - mask) * tgt
        res = res.astype(np.uint8)
        return res



class MultiBandBlend(Blend):
    def __init__(self, params):
        super().__init__()
        self.depth = params['depth']

    def get_mask(self,shape):
        rough_mask = np.zeros(shape, dtype=np.uint8)
        rough_mask[shape[0]//2-100:shape[0]//2+100, shape[1]//2-100:shape[1]//2+100] = 255
        rough_mask = rough_mask.astype(np.uint8)
        return rough_mask

    def resize_shape(self,img):
        new_shape = (img.shape[1] & ~(2**(self.depth-1)-1), img.shape[0] & ~(2**(self.depth-1)-1))
        return  cv2.resize(img, new_shape)

    
    def build_gaussian_pyramid(self,img):
        pyramid = [img]
        for i in range(self.depth-1):
            img = cv2.pyrDown(img)
            pyramid.append(img.astype(np.uint8))
        return pyramid

    def build_laplacian_pyramid(self,gaussian_pyramid):
        laplacian_pyramid = [gaussian_pyramid[-1]]
        for i in range(self.depth-1, 0, -1):
            gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
            laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
            laplacian_pyramid.append(laplacian.astype(np.uint8))

        return laplacian_pyramid
    
    def reconstruct_img(self,blended_pyramid):
        img = blended_pyramid[0]
        for i in range(1,len(blended_pyramid)):
            img = cv2.pyrUp(img)
            img = cv2.add(img, blended_pyramid[i])
        return img


    def blend(self, src, tgt, mask=None):
        self.mask = self.get_mask() if mask is None else mask
        src, tgt, mask = self.resize_shape(src), self.resize_shape(tgt), self.resize_shape(mask)
        # Build the Gaussian pyramids
        gaussian_pyramid1 = self.build_gaussian_pyramid(src)
        gaussian_pyramid2 = self.build_gaussian_pyramid(tgt)
        mask_gaussian_pyramid = self.build_gaussian_pyramid(mask)

        # Build the Laplacian pyramids
        laplacian_pyramid1 = self.build_laplacian_pyramid(gaussian_pyramid1)
        laplacian_pyramid2 = self.build_laplacian_pyramid(gaussian_pyramid2)


        # Blend the images
        blended_pyramid = []
        for i in range(self.depth-1,-1,-1):
            mask_gaussian  = mask_gaussian_pyramid[i]/255
            mask_gaussian = mask_gaussian / np.max(mask_gaussian)
            im1 = laplacian_pyramid1[self.depth-i-1].astype(float)
            im2 = laplacian_pyramid2[self.depth-i-1].astype(float)
            k1 = np.multiply(im1, mask_gaussian).astype(np.uint8)
            k2 = np.multiply((1-mask_gaussian), im2).astype(np.uint8)
            level_blend = k1.astype(np.uint8) + k2.astype(np.uint8)
            blended_pyramid.append(level_blend.astype(np.uint8))


        res = self.reconstruct_img(blended_pyramid)
        return res
        

class PoissonBlend(Blend):
    def __init__(self, params):
        super().__init__()
        self.mode = params['mode']
        self.solver = params['solver']
        self.alpha = params['alpha'] if 'alpha' in params else 0
        self.solver_func = getattr(scipy.sparse.linalg, self.solver) if self.solver != "multigrid" else None


    def get_mask(self,img):
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img.shape[0]//2-50:img.shape[0]//2+50, img.shape[1]//2-50:img.shape[1]//2+50] = 255
        return mask

    def process_mask(self,mask):
        boundary = find_boundaries(mask, mode="inner").astype(int)
        inner = mask - boundary
        return inner, boundary
    
    def get_pixel_ids(self,img):
        pixel_ids = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1])
        return pixel_ids
    
    
    def get_masked_values(self,values,mask):
        assert values.shape == mask.shape
        return values[np.nonzero(mask)]
    
    def get_alpha_blended_img(self,src,tgt,alpha_mask):
        return src * alpha_mask + tgt * (1 - alpha_mask)
    
    def construct_A_matrix(self,img_shape,mask_ids,inner_ids,inner_pos,boundary_pos):
        width = img_shape[1]

        n1_pos = np.searchsorted(mask_ids, inner_ids - 1)
        n2_pos = np.searchsorted(mask_ids, inner_ids + 1)
        n3_pos = np.searchsorted(mask_ids, inner_ids - width)
        n4_pos = np.searchsorted(mask_ids, inner_ids + width)

        A = scipy.sparse.lil_matrix((len(mask_ids), len(mask_ids)))
        A[inner_pos, n1_pos] = 1
        A[inner_pos, n2_pos] = 1
        A[inner_pos, n3_pos] = 1
        A[inner_pos, n4_pos] = 1
        A[inner_pos, inner_pos] = -4 
        A[boundary_pos, boundary_pos] = 1
        A = A.tocsr()
        return A
    
    def construct_b_vector(self,mask_ids,inner_pos,boundary_pos,inner_gradient_values, boundary_pixel_values):
        b = np.zeros(len(mask_ids))
        b[inner_pos] = inner_gradient_values
        b[boundary_pos] = boundary_pixel_values
        return b
    
    def compute_gradient(self,img,forward = True):
        if forward:
            kx = np.array([[0, 0, 0],[0, -1, 1],[0, 0, 0]])
            ky = np.array([[0, 0, 0],[0, -1, 0],[0, 1, 0]])
        else:
            kx = np.array([[0, 0, 0],[-1, 1, 0],[0, 0, 0]])
            ky = np.array([[0, -1, 0],[0, 1, 0],[0, 0, 0]])
        Gx = scipy.signal.fftconvolve(img, kx, mode="same")
        Gy = scipy.signal.fftconvolve(img, ky, mode="same")
        return Gx, Gy
    
    def compute_laplacian(self,img):
        kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
        laplacian = scipy.signal.fftconvolve(img, kernel, mode="same")
        return laplacian
    
    def calculate_mixed_gradient(self,src,tgt):
        if self.mode == 'max':
            Ix_src, Iy_src = self.compute_gradient(src)
            Ix_target, Iy_target = self.compute_gradient(tgt)
            I_src_amp = (Ix_src**2 + Iy_src**2)**0.5
            I_target_amp = (Ix_target**2 + Iy_target**2)**0.5
            Ix = np.where(I_src_amp > I_target_amp, Ix_src, Ix_target)
            Iy = np.where(I_src_amp > I_target_amp, Iy_src, Iy_target)
            Ixx, _ = self.compute_gradient(Ix, forward=False)
            _, Iyy = self.compute_gradient(Iy, forward=False)

            return Ixx + Iyy
        elif self.mode == 'alpha':
            src_laplacian = self.compute_laplacian(src)
            target_laplacian = self.compute_laplacian(tgt)
            return self.alpha * src_laplacian + (1 - self.alpha) * target_laplacian
        
    def poisson_blend_channel(self,src,tgt,mask,boundary_mask,inner_mask,mask_pos,inner_pos,boundary_pos,mask_ids,A):
        mixed_gradient = self.calculate_mixed_gradient(src, tgt)
        boundary_pixel_values = self.get_masked_values(tgt, boundary_mask).flatten()
        inner_gradient_values = self.get_masked_values(mixed_gradient, inner_mask).flatten()

        b = self.construct_b_vector(mask_ids,inner_pos,boundary_pos,inner_gradient_values, boundary_pixel_values)

        if self.solver == "multigrid":
            x = self.solver_func(A, b)
            if isinstance(x, tuple):
                x = x[0]
        else:
            ml = pyamg.ruge_stuben_solver(A)
            x = ml.solve(b, tol=1e-10)

        new_src = np.zeros(src.size)
        new_src[mask_pos] = x
        new_src = new_src.reshape(src.shape)
        poisson_blended_img = self.get_alpha_blended_img(new_src, tgt, mask)
        poisson_blended_img = np.clip(poisson_blended_img, 0, 1)
    
        return poisson_blended_img
    
    def blend(self,src,tgt,mask =None):
        self.mask = self.get_mask() if mask is None else mask
        mask = cv2.cvtColor(self.mask,  cv2.COLOR_RGB2GRAY)
        mask = mask.astype(np.float32) / 255
        src, tgt = src.astype(float)/255, tgt.astype(float)/255

        _, mask = cv2.threshold(mask, 0.5,1, cv2.THRESH_BINARY)
        inner_mask , boundary_mask = self.process_mask(mask)
        pixel_ids = self.get_pixel_ids(mask)
        inner_ids = self.get_masked_values(pixel_ids, inner_mask).flatten()
        boundary_ids = self.get_masked_values(pixel_ids, boundary_mask).flatten()
        mask_ids = self.get_masked_values(pixel_ids, mask).flatten()

        inner_pos = np.searchsorted(mask_ids, inner_ids)
        boundary_pos = np.searchsorted(mask_ids, boundary_ids)
        mask_pos = np.searchsorted(pixel_ids.flatten(), mask_ids)

        A = self.construct_A_matrix(src.shape,mask_ids,inner_ids,inner_pos,boundary_pos)

        poisson_blended_img_rgb = []
        for i in range(src.shape[-1]):         
            poisson_blended_img_rgb.append(self.poisson_blend_channel(src[:,:,i],tgt[:,:,i],mask,boundary_mask,inner_mask,mask_pos,inner_pos,boundary_pos,mask_ids,A))
        
        poisson_blended_img = np.dstack(poisson_blended_img_rgb)
        poisson_blended_img = (poisson_blended_img*255).astype(np.uint8)
        return poisson_blended_img


class PoissonBlendInBuilt(Blend):
    def __init__(self, params):
        super().__init__()
        #self.center = params['center']
        pass 
    def get_mask(self,img):
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img.shape[0]//2-50:img.shape[0]//2+50, img.shape[1]//2-50:img.shape[1]//2+50] = 255
        
        rough_mask = np.zeros(img.shape, dtype=np.uint8)
        rough_mask[img.shape[0]//2-100:img.shape[0]//2+100, img.shape[1]//2-100:img.shape[1]//2+100] = 255

        return mask, rough_mask

    def get_center(self,rough_mask):
        # Use the rough mask to find the center of the image
        M = cv2.moments(rough_mask[:,:,0])
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center

    def blend(self,src,tgt,mask =None):
        if mask is None:
            mask, rough_mask = self.get_mask(src)
            center = self.get_center(rough_mask)
        else:
            center = self.get_center(mask)

        return cv2.seamlessClone(src, tgt, mask, center, cv2.NORMAL_CLONE)


def create_blending_technique(technique_name, params):
    if technique_name == 'CutPaste':
        return CutPaste(params)
    elif technique_name == 'Alpha':
        return AlphaBlend(params)
    elif technique_name == 'MultiBand':
        return MultiBandBlend(params)
    elif technique_name == 'Poisson':
        return PoissonBlend(params)
    elif technique_name == 'Poisson_InBuilt':
        return PoissonBlendInBuilt(params)


if __name__ == "__main__":
    
    print("Select a blending technique::\n1. CutPaste\n2. Alpha\n3. MultiBand\n4. Poisson_Scratch\n5.Poisson_InBuilt")
    choice = int(input("Enter your choice (1-5): "))
    params = {}
    if choice == 1:
        blending_technique = create_blending_technique('CutPaste', params)
    elif choice == 2:
        kernel_size = int(input("Enter the kernel size for Alpha blending: "))
        params['kernel'] = (kernel_size,kernel_size)
        blending_technique = create_blending_technique('Alpha', params)
    elif choice == 3:
        depth = int(input("Enter the depth for MultiBand blending: "))
        params['depth'] = depth
        blending_technique = create_blending_technique('MultiBand', params)
    elif choice == 4:
        mode = input("Enter the mode for Poisson blending (max/alpha): ")
        params['mode'] = mode
        if mode == 'alpha':
            alpha = float(input("Enter the alpha value for Poisson blending: "))
            params['alpha'] = alpha
        solver = input("Enter the solver for Poisson blending: ")
        params['solver'] = solver
        blending_technique = create_blending_technique('Poisson', params)
    elif choice == 5:
        blending_technique = create_blending_technique('Poisson_InBuilt', params)

    dir = 'D:/PDFs/4th year-2nd sem/CV/computer_vision_project/test'
    out_dir = 'D:/PDFs/4th year-2nd sem/CV/computer_vision_project/output/'
    source_path = path.join(dir, "source.png")
    destination_path = path.join(dir, "target.png")
    mask_path = path.join(dir, "mask.png")
    src_img = np.array(Image.open(source_path))
    dest_img = np.array(Image.open(destination_path))
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask,  cv2.COLOR_BGR2RGB)
    blended_data = blending_technique.blend(src_img, dest_img, mask)
    if choice == 4:
        output_path = path.join(out_dir, 'blended_img_'+str(choice)+'_'+mode+'.png')
    else:
        output_path = path.join(out_dir, 'blended_img_'+str(choice)+'.png')

    plt.imsave(output_path, blended_data)
    print("Blended image saved at: ", output_path)

