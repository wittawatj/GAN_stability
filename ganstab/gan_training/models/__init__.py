from ganstab.gan_training.models import (
    resnet, resnet2,resnet4, 
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'resnet4': resnet4.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet4' : resnet4.Discriminator,
}
