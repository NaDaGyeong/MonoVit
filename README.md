# MonoVit

## Train
PLease download the ImageNet-1K pretrained MPViT model(https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth) to ./ckpt/.

For training, please download monodepth2, replace the depth network, and revise the setting of the depth network, the optimizer and learning rate according to trainer.py.

Because of the different torch version between MonoViT and Monodepth2, the func transforms.ColorJitter.get_params in dataloader should also be revised to transforms.ColorJitter.

By default models and tensorboard event files are saved to ./tmp/<model_name>. This can be changed with the --log_dir flag.
