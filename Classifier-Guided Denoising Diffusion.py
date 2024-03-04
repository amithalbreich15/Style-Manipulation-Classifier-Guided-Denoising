### EX4 PART 2 - SUBMITTED BY: ###

### AMIT HALBREICH & EITAN STERN ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

matplotlib.use("TkAgg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./picasso.jpg")
content_img = image_loader("./dancing.jpg")

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class MeanVarStyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(MeanVarStyleLoss, self).__init__()
        self.target_mean = torch.mean(target_feature, dim=(2, 3), keepdim=True)
        self.target_var = torch.var(target_feature, dim=(2, 3), keepdim=True)

    def forward(self, input):
        input_mean = torch.mean(input, dim=(2, 3), keepdim=True)
        input_var = torch.var(input, dim=(2, 3), keepdim=True)

        mean_loss = F.mse_loss(input_mean, self.target_mean)
        var_loss = F.mse_loss(input_var, self.target_var)

        self.mean_var_loss = mean_loss + var_loss

        return input


cnn = models.vgg19(pretrained=True).features.eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is n umber of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_loss_func,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default
                               ):
    # Normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # Just in order to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []

    # Assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # Increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError(
                'Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # Add content loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Add style loss
            target_feature = model(style_img).detach()
            if style_loss_func == 'gram':
                style_loss = StyleLoss(target_feature)
                style_losses.append(style_loss)
            elif style_loss_func == 'mean_var':
                style_loss = MeanVarStyleLoss(target_feature)
                style_losses.append(style_loss)
            else:
                raise ValueError(
                    'Invalid style loss function: {}'.format(style_loss_func))
            model.add_module("style_loss_{}".format(i), style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i],
                                                           StyleLoss) or isinstance(
                model[i], MeanVarStyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1e6, content_weight=1, content_layers=content_layers_default,
                       style_layers=style_layers_default, style_loss_func='gram'):
    """Run the style transfer."""
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std,
                                                                     style_img, content_img, content_layers,
                                                                     style_layers,style_loss_func)
    style_loss_list = []
    content_loss_list = []

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # Correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                if style_loss_func == 'gram':
                    style_score += sl.loss
                elif style_loss_func == 'mean_var':
                    style_score += sl.mean_var_loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                if style_loss_func == 'mean_var':
                    print('MeanVar Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                elif style_loss_func == 'gram':
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                print()
                style_loss_list.append(style_score.item())
                content_loss_list.append(content_score.item())

            # Update the losses in a separate list
            closure.style_loss_list = style_loss_list
            closure.content_loss_list = content_loss_list

            return style_score + content_score

        optimizer.step(closure)

    # A last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img, style_loss_list, content_loss_list



from torchvision.models.vgg import vgg16_bn

content_layers_def = [['conv_1'], ['conv_5'], ['conv_10'], ['relu_1'],
                      ['relu_5'], ['relu_10'], ['pool_2'], ['pool_8'],
                      ['pool_16']]
style_layers_def = [['conv_1'], ['conv_5'], ['conv_10'], ['relu_1'],
                    ['relu_5'], ['relu_10'], ['pool_2'], ['pool_8'],
                    ['pool_16']]


def feature_inversion(normalization_mean, normalization_std,
                      content_img, num_steps=1000,
                      style_weight=0, content_weight=1,
                      content_layers=content_layers_def,
                      style_layers=style_layers_def, output_path='output.jpg'):
    output_images = []
    print(len(content_layers))
    for i in range(len(content_layers)):
        # style_layers[i]
        style_layer = []
        content_layer = content_layers[i]
        # Define the optimizer
        vgg = models.vgg19(pretrained=True).features.eval()

        input_img = torch.randn(content_img.size(), device=device,
                                requires_grad=True)

        # # We want to optimize the input and not the model parameters so we
        # # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)

        print(content_layer)
        output_image, style_losses, content_losses = run_style_transfer(vgg,
                                                                        cnn_normalization_mean,
                                                                        cnn_normalization_std,
                                                                        content_img,
                                                                        input_img,
                                                                        input_img,
                                                                        style_weight=0,
                                                                        content_weight=1,
                                                                        content_layers=content_layer,
                                                                        style_layers=style_layer)
        output_images.append(output_image)
    return output_images


from torchvision.models.vgg import vgg16_bn

content_layers_def = [['conv_1'], ['conv_5'], ['conv_10'], ['relu_1'],
                      ['relu_5'], ['relu_10'], ['pool_2'], ['pool_8'],
                      ['pool_16']]
style_layers_def = [['conv_1'], ['conv_5'], ['conv_10'], ['relu_1'],
                    ['relu_5'], ['relu_10'], ['pool_2'], ['pool_8'],
                    ['pool_16']]


def texture_synthesis(normalization_mean, normalization_std,
                      style_img, num_steps=300,
                      style_weight=1e6, content_weight=0,
                      content_layers=content_layers_def,
                      style_layers=style_layers_def, output_path='output.jpg'):
    output_images = []
    print(len(style_layers))
    for i in range(len(style_layers)):
        # style_layers[i]
        style_layer = style_layers[i]
        # content_layer = content_layers[i]
        content_layer = []
        # Define the optimizer
        vgg = models.vgg19(pretrained=True).features.eval()

        input_img = torch.randn(style_img.size(), device=device,
                                requires_grad=True)

        # # We want to optimize the input and not the model parameters so we
        # # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)

        print(content_layer)
        output_image, style_losses, content_losses = run_style_transfer(vgg,
                                                                        cnn_normalization_mean,
                                                                        cnn_normalization_std,
                                                                        input_img,
                                                                        style_img,
                                                                        input_img,
                                                                        style_weight=1e6,
                                                                        content_weight=0,
                                                                        content_layers=content_layer,
                                                                        style_layers=style_layer)
        output_images.append(output_image)
    return output_images


import matplotlib.pyplot as plt
import numpy as np


def plot_batch_images(images):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.tight_layout()

    titles = ['conv_1', 'conv_5', 'conv_10', 'relu_1', 'relu_5', 'relu_10',
              'pool_2', 'pool_8', 'pool_16']

    for i, ax in enumerate(axes.flat):
        image = np.transpose(images[i],
                             (1, 2, 0))  # Reshape to (height, width, channels)
        ax.imshow(image)
        ax.set_title(titles[i])
        ax.axis('off')

    plt.show()


### PART 2 CODE ###
from torch import optim
from torchvision import models, transforms
from torch.cuda import empty_cache
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, \
    UniPCMultistepScheduler
import torchvision.models as tmodels
from tqdm import tqdm


def load_components():
    subfolder = "CompVis/stable-diffusion-v1-4"
    return {
        'vae': AutoencoderKL.from_pretrained(subfolder, subfolder="vae"),
        'tokenizer': CLIPTokenizer.from_pretrained(subfolder,
                                                   subfolder="tokenizer"),
        'text_encoder': CLIPTextModel.from_pretrained(subfolder,
                                                      subfolder="text_encoder"),
        'unet': UNet2DConditionModel.from_pretrained(subfolder,
                                                     subfolder="unet"),
        'scheduler': UniPCMultistepScheduler.from_pretrained(subfolder,
                                                             subfolder="scheduler")
    }


def image_load(image_name):
    imsize = 512 if torch.cuda.is_available() else 128
    transform = transforms.Compose(
        [transforms.Resize(imsize), transforms.ToTensor()])
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


def process_text(prompt=["a cat"], steps=150, scale=7.5, s=0.6,
                 style_img1="picasso.jpg"):
    latents_list = []
    height = width = 512
    batch_size = len(prompt)
    models = load_components()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_input = models['tokenizer'](prompt, padding="max_length",
                                     max_length=models[
                                         'tokenizer'].model_max_length,
                                     truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = \
        models['text_encoder'](text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = models['tokenizer']([""] * batch_size, padding="max_length",
                                       max_length=max_length,
                                       return_tensors="pt")
    uncond_embeddings = \
    models['text_encoder'](uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn((batch_size, models['unet'].config.in_channels,
                           height // 8, width // 8)).to(device)

    latents = latents * models['scheduler'].init_noise_sigma

    models['scheduler'].set_timesteps(steps)
    style_img1 = image_load(style_img1)
    style_loss_func = 'gram'  # Change to 'mean_var' to do StyleGuidance
    # according to mean_var style loss
    cnn = tmodels.vgg19(pretrained=True).features.to(device).eval()
    style_model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                           cnn_normalization_mean,
                                                                           cnn_normalization_std,
                                                                           style_img1,
                                                                           content_img,
                                                                           style_loss_func)

    for i, t in enumerate(tqdm(models['scheduler'].timesteps)):
        latent_input = torch.cat([latents] * 2)
        latent_input = models['scheduler'].scale_model_input(latent_input,
                                                             timestep=t)

        with torch.no_grad():
            noise_pred = models['unet'](latent_input, t,
                                        encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + scale * (
                    noise_pred_text - noise_pred_uncond)

        z = torch.tensor(noise_pred.clone().detach(), requires_grad=True,
                         device=device)
        clean_pred = 1 / 0.18215 * models['scheduler'].convert_model_output(z,
                                                                            t,
                                                                            latents)

        decoded = models['vae'].decode(clean_pred).sample
        style_model(decoded / 2 + 0.5)

        style_score = 0
        for sl in style_losses:
            if style_loss_func == 'gram':
                style_score += sl.loss
            elif style_loss_func == 'mean_var':
                style_score += sl.mean_var_loss

        loss = style_score
        loss.backward()

        with torch.no_grad():
            grad = z.grad / torch.mean(torch.abs(z.grad))
        latents = models['scheduler'].step(
            noise_pred - s * grad * models['scheduler'].sigma_t[t], t,
            latents).prev_sample
        torch.cuda.empty_cache()

        if (i + 1) % 25 == 0:
            latents_list.append(latents)
    return latents_list


def create_image_from_latents(latents_list, output_file='output.png'):
    latents = latents_list[5]
    latents *= 1 / 0.18215

    models = load_components()

    with torch.no_grad():
        decoded = models['vae'].decode(latents).sample
    decoded_image = (decoded / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0,
                                                                           2,
                                                                           3,
                                                                           1).numpy()
    Image.fromarray((decoded_image * 255).round().astype("uint8")[0]).save(
        output_file)


latents_list = process_text()
create_image_from_latents(latents_list)
