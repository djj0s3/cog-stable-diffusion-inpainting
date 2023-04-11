import os
from typing import List

import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
import PIL.ImageOps
from cog import BasePredictor, Input, Path

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


# MODEL_ID refers to a diffusers-compatible model on HuggingFace
# e.g. prompthero/openjourney-v2, wavymulder/Analog-Diffusion, etc
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        negative_prompt: str = Input(description="The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).", default=""),
        image: Path = Input(
            description="Input image to in-paint. Width and height should both be divisible by 8. If they're not, the image will be center cropped to the nearest width and height divisible by 8",
        ),
        mask: Path = Input(
            description="Black and white image to use as mask. White pixels are inpainted and black pixels are preserved.",
        ),
        invert_mask: bool = Input(
            description="If this is true, then black pixels are inpainted and white pixels are preserved.",
            default=False,
        ),
        num_outputs: int = Input(
            description="Number of images to output. NSFW filter in enabled, so you may get fewer outputs than requested if flagged",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(image).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        if invert_mask:
            mask = PIL.ImageOps.invert(mask)

        if image.width % 8 != 0 or image.height % 8 != 0:
            if mask.size == image.size:
                mask = crop(mask)
            image = crop(image)

        if mask.size != image.size:
            print(
                f"WARNING: Mask size ({mask.width}, {mask.height}) is different to image size ({image.width}, {image.height}). Mask will be resized to image size."
            )
            mask = mask.resize(image.size)
            
        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_images_per_prompt=num_outputs,
            mask_image=mask,
            width=image.width,
            height=image.height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def crop(image):
    height = (image.height // 8) * 8
    width = (image.width // 8) * 8
    left = int((image.width - width) / 2)
    right = left + width
    top = int((image.height - height) / 2)
    bottom = top + height
    image = image.crop((left, top, right, bottom))
    return image

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]