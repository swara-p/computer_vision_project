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
import torch
import torch.optim as optim

from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, \
                  MeanShift, Vgg16, gram_matrix


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

class DeepBlend(Blend):
    def __init__(self, params):
        super().__init__()
        self.num_steps = params['num_steps']
        self.grad_weight = 1e4 
        self.style_weight = 1e4 
        self.content_weight = 1
        self.tv_weight = 1e-6
        self.gpu_id = 0
        self.mse = torch.nn.MSELoss()
        self.mean_shift = MeanShift(self.gpu_id )
        self.vgg = Vgg16().to(self.gpu_id )
    
    def get_mask(self,img):
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img.shape[0]//2-50:img.shape[0]//2+50, img.shape[1]//2-50:img.shape[1]//2+50] = 255
        return mask
    
    def numpy_images(self,src,tgt,mask):
        src = torch.from_numpy(src).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(0)
        tgt = torch.from_numpy(tgt).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(0)
        inpt_img = torch.randn(tgt.shape).to(0)

        mask_img = numpy2tensor(mask,0)
        mask_img = mask_img.squeeze(0).repeat(3,1).view(3,src.shape[2],src.shape[3]).unsqueeze(0)

        return src, tgt, inpt_img, mask_img
    
    def get_optimzer(self,inpt_img):
        optimizer = optim.LBFGS([inpt_img.requires_grad_()])
        return optimizer

    def first_pass(self,src,tgt,mask_img,canvas_mask,inpt_img,blend_img,true_grad,optimizer):
        run = [0]
        x_start, y_start = src.shape[2]//2, src.shape[3]//2
        print('Entering first pass')
        while run[0] <= self.num_steps:
            def closure():
                blended_img = torch.zeros(tgt.shape).to(0)
                blended_img = inpt_img * canvas_mask + tgt * (1 - canvas_mask)

                grad = laplacian_filter_tensor(blended_img,0)

                grad_loss = 0
                for i in range(len(grad)):
                    grad_loss += self.mse(grad[i],true_grad[i])
                grad_loss = grad_loss / len(grad)
                grad_loss = grad_loss * self.grad_weight

                # Style Loss
                tgt_feature_style_loss =  self.vgg(self.mean_shift(tgt))
                tgt_feature_style_gram = [gram_matrix(y) for y in tgt_feature_style_loss]

                blend_feat_style_loss = self.vgg(self.mean_shift(inpt_img))
                blend_feat_style_gram = [gram_matrix(y) for y in blend_feat_style_loss]

                style_loss = 0
                for i in range(len(blend_feat_style_gram)):
                    style_loss += self.mse(blend_feat_style_gram[i],tgt_feature_style_gram[i])
                style_loss = style_loss / len(blend_feat_style_gram)
                style_loss = style_loss * self.style_weight

                bended_obj = blend_img[:,:,int(x_start-src.shape[2]*0.5):int(x_start+src.shape[2]*0.5), int(y_start-src.shape[3]*0.5):int(y_start+src.shape[3]*0.5)]
                src_obj_feat = self.vgg(self.mean_shift(src*mask_img))
                blend_obj_feat = self.vgg(self.mean_shift(bended_obj*mask_img))
                contnet_loss = self.content_weight*self.mse(blend_obj_feat.relu2_2,src_obj_feat.relu2_2)
                contnet_loss = contnet_loss * self.content_weight

                tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                   torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
                tv_loss *= self.tv_weight

                loss = grad_loss + style_loss + contnet_loss + tv_loss
                optimizer.zero_grad()
                loss.backward()

                if run[0]%100== 0:
                    print("Step: ", run[0], "grad_loss: ", grad_loss.item(), "style_loss: ", style_loss.item(), "content_loss: ", contnet_loss.item(), "tv_loss: ", tv_loss.item())
                
                run[0] += 1
                return loss
            
            optimizer.step(closure)

        inpt_img.data.clamp_(0, 255)

        blended_img = torch.zeros(tgt.shape).to(self.gpu_id)
        blended_img = inpt_img * canvas_mask + tgt * (1 - canvas_mask)
        blended_img_np = blended_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

        return blended_img_np.astype(np.uint8)
    

    def second_pass(self,first_pass_img,tgt):
        print('Entering second pass: Optimizing the blended image')
        self.weight = 1e7
        first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(self.gpu_id)
        first_pass_img = first_pass_img.contiguous()
        tgt = tgt.contiguous()
        optimizer = self.get_optimzer(first_pass_img)
        run = [0]
        while run[0] <= self.num_steps:
            def closure():
                tgt_feat_style = self.vgg(self.mean_shift(tgt))
                tgt_gram = [gram_matrix(y) for y in tgt_feat_style]
                blended_feat_style = self.vgg(self.mean_shift(first_pass_img))
                blend_gram = [gram_matrix(y) for y in blended_feat_style]

                style_loss = 0
                for i in range(len(blend_gram)):
                    style_loss += self.mse(blend_gram[i], tgt_gram[i])
                style_loss = style_loss / len(blend_gram)
                style_loss = style_loss * self.style_weight

                content_feat = self.vgg(self.mean_shift(first_pass_img))
                content_loss = self.content_weight*self.mse(blended_feat_style.relu2_2, content_feat.relu2_2)

                loss = style_loss + content_loss
                optimizer.zero_grad()
                loss.backward()

                if run[0]%100 == 0:
                    print("Step: ", run[0], "style_loss: ", style_loss.item(), "content_loss: ", content_loss.item())
                
                run[0] += 1
                return loss
            optimizer.step(closure)
        
        first_pass_img.data.clamp_(0, 255)
        final_img = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]
        return final_img.astype(np.uint8)


    def blend(self,src,tgt,mask =None):
        self.mask = self.get_mask() if mask is None else mask
        mask_img = cv2.cvtColor(mask,  cv2.COLOR_RGB2GRAY)
        mask_img[mask_img>0] = 1
        src_shape = src.shape
        tgt_shape = tgt.shape
        x_start, y_start = src_shape[0]//2, src_shape[1]//2
        
        canvas_mask = make_canvas_mask(x_start,y_start, tgt, mask_img)
        canvas_mask = numpy2tensor(canvas_mask,self.gpu_id)
        canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3,tgt_shape[0],tgt_shape[1]).unsqueeze(0)

        true_gradient = compute_gt_gradient(x_start, y_start, src, tgt, mask_img,0)
        
        src, tgt, inpt_img, mask_img = self.numpy_images(src,tgt,mask_img)

        optimizer = self.get_optimzer(inpt_img)

        first_pass_img = self.first_pass(src,tgt,mask_img,canvas_mask,inpt_img,src,true_gradient,optimizer)

        second_pass_img = self.second_pass(first_pass_img,tgt)

        return second_pass_img
    


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
    elif technique_name == 'DeepBlend':
        return DeepBlend(params)


if __name__ == "__main__":
    
    print("Select a blending technique::\n1. CutPaste\n2. Alpha\n3. MultiBand\n4. Poisson_Scratch\n5.Poisson_InBuilt\n6. DeepBlend\n")
    choice = int(input("Enter your choice (1-6): "))
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
    elif choice == 6:
        num_steps = int(input("Enter the number of steps for DeepBlend: "))
        params['num_steps'] = num_steps
        blending_technique = create_blending_technique('DeepBlend', params)

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

