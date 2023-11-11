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
                - in: b c f h w
                - proc: reshape to b*f c h w -> conv -> reshape back
                - out: b c f h w
            -down_blocks
                -CrossAttnDownBlock3D x 3
                    -ResBlock3D
                        - in: x: (b c f h w), temb: (b dt)
                        - out: (b c f h w)
                        - proc: 
                            - GN: support two types
                                - normal: (b c f h w) -> GN -> (b*f c h w)  # mean on whole whole -> inference_v1.yaml, **use_inflated_groupnorm**=False
                                - inflated: (b c f h w) -> (b*f c h w) -> GN -> (b*f c h w) -> (b c f h w)  # mean on each frame, -> inference_v2.yaml, True
                            - SiLU 
                            - InflatedConv 
                                - reshape to bxf ..., conv, reshape back
                            - GN
                            - SiLU, dropout, InflatedConv
                    -SpatialTransformer3D
                        - input: x (b c f h w), context (b 77 768)
                        - x -> (b*f c h w) -> GN -> (b*f h w c) -> (b*f h*w c)  
                        - context -> (b*f 77 768)
                        - BasicTransformer:
                            - CrossAttention: 
                            - LayerNorm
                            - FF 
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
    
