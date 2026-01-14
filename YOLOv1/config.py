from dataclasses import dataclass

#defining the architecture configuration from the official paper
@dataclass
class config:
    architecture_config = [
        ##(kernel_size,out_channels,stride,paading)
        (7,64,2,3),
        "max_pool",
        (3,192,1,1),
        "max_pool",
        (1,128,1,0),
        (3,256,1,1),
        (1,256,1,0),
        (3,512,1,1),
        "max_pool",
        [
            ##4 times internally repeated convolutional block
            (1,256,1,0),
            (3,512,1,1),
            4
        ],
        (1,512,1,0),
        (3,1024,1,1),
        "max_pool",
        [
            ## 2 times internally repeated convolutional block
            (1,512,1,0),
            (3,1024,1,1),
            2
        ],
        (3,1024,1,1),
        (3,1024,2,1),
        (3,1024,1,1),
        (3,1024,1,1),
    ]