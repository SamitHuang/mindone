MS-optimized AnimateDiff

Since torch AD relies heavily on diffusers and transformers, we will build a new code framework.
    1. SD modules are from LDM, base config: ../stable_diffusion_v2/configs/v1-inference.yaml

## Design Rules
1. Build framework. Then get what we want from those sources. 
2. Double check input-output when using existing modules.
3. user-end api, make it as closer to original AD as possible

## TODO: 
1. Code framework 
    - setup code file structure 
    - setup code key API structure
    - config method:
        Need to merge sd_v1_inferernce.yaml with ad_inference_v1.yaml
    
2. Net Modules
    - CLIP-vit l/14 text encoder 
        - API: input config, return clip 
        [ ] transfer from SD 
        [ ] check: padding token! 
    - VAE 
    - UNet 3D
        - Basic Structure
            -conv_in = InflatedConv
                - in: b d f h w
                - proc: reshape to b*f d h w -> conv -> reshape back
                - out: b d f h w
            -down_blocks
                -CrossAttnDownBlock3D x 3
                    -ResBlock3D
                        - in: (b d f h w)
                        - out: (b d f h w)
                        - proc: 
                            - GN [so... ch order matters], SiLU, 
                            - InflatedConv  
                                - reshape to bxf ..., conv, reshape back
                    -SpatialTransformer3D 
                        - input: 
                        - output: 
                    -MotionModuel /TemporalTransformer 
                -DownBlock3D
            -middle_block
            -up_block
        - lift 2d to pseudo 3d
        - input for  

3. Inference pipeline
    - Basic t2i
    - 
    
