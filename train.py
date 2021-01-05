from tqdm import trange
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import imageio
from skimage.transform import resize
from skimage import img_as_ubyte

from demo import make_animation
from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater

import subprocess

def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, opt):
    device_ids = opt.device_ids
    
    print("Using devices: ", device_ids)
    print("Logging to: ", log_dir)
    train_params = config['train_params']
    dataset_params = config['dataset_params']
    
    # Check if we are testing the model during training on a driving video and source image.
    # If so, load them and resize them to proper dimensions based on *.yaml config file used for training.
    #
    source_image = None
    driving_video = None
    if (opt.driving_vid is not None) and (opt.source_img is not None):
        print("Using driving video and source image to test model during training")
        source_image = imageio.imread(opt.source_img)
        reader = imageio.get_reader(opt.driving_vid)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            print("!!! Error: failed to load frames from driving video")
            pass
        reader.close()
        
        frame_shape = dataset_params['frame_shape']
        frame_dims = (frame_shape[0], frame_shape[1])
        print("\tResizing video and image to: ", frame_dims)
        source_image = resize(source_image, frame_dims)[..., :3]
        driving_video = [resize(frame, frame_dims)[..., :3] for frame in driving_video] 

        
    optimizer_generator = torch.optim.Adam(generator.parameters(),
                                           lr=train_params['lr_generator'],
                                           betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),
                                               lr=train_params['lr_discriminator'],
                                               betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(),
                                             lr=train_params['lr_kp_detector'],
                                             betas=(0.5, 0.999))

    if checkpoint is not None:
        print("Using checkpoint: ", checkpoint)
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0
        
    print("starting training from epoch: ", start_epoch)

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            x = None
            generated = None
            for x in dataloader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            if (epoch % 10) == 0:
                print("EPOCH: ", epoch)
                print("Logging and saving checkpoint... ", log_dir)
                logger.log_epoch_and_checkpoint(epoch, 
                                                {'generator': generator,
                                                 'discriminator': discriminator,
                                                 'kp_detector': kp_detector,
                                                 'optimizer_generator': optimizer_generator,
                                                 'optimizer_discriminator': optimizer_discriminator,
                                                 'optimizer_kp_detector': optimizer_kp_detector},
                                                inp=x, out=generated)
                
                if (driving_video is not None) and (source_image is not None):
                    predictions = make_animation(source_image,
                                                 driving_video,
                                                 generator,
                                                 kp_detector,
                                                 relative=False,
                                                 adapt_movement_scale=False,
                                                 cpu=False)
                    result_vid = f'{log_dir}/result_epoch{epoch}.mp4'
#                     print("Saving result video: ", result_vid)
                    imageio.mimsave(result_vid, [img_as_ubyte(frame) for frame in predictions], fps=fps)
                    
#                 #Generate synthesized video 
#                 drivingvideo = '/home/jupyter/test_video_youtube/512output.mkv'
#                 source_image = '/home/jupyter/test_video_youtube/first_frame.png'
#                 checkpointpath = os.path.join(log_dir, '%s-checkpoint.pth.tar' % str(epoch).zfill(self.zfill_num)) 
#                 res = '/home/jupyter/test_video_youtube/results/' + epoch + '.mp4'
#                 cmd = ["python","./demo.py","--config", "config/vox-512.yaml", "--driving_video", drivingvideo, 
#                        "--source_image", source_image, "--checkpoint", checkpointpath, "--relative", "--adapt_scale","--result_video", res]
#                 subprocess.Popen(cmd).wait()
        
        # Ensure final epoch is saved.
        #
        print("EPOCH: ", epoch)
        print("Logging and saving checkpoint... ", log_dir)
        logger.log_epoch_and_checkpoint(epoch, 
                                        {'generator': generator,
                                         'discriminator': discriminator,
                                         'kp_detector': kp_detector,
                                         'optimizer_generator': optimizer_generator,
                                         'optimizer_discriminator': optimizer_discriminator,
                                         'optimizer_kp_detector': optimizer_kp_detector},
                                        inp=x, out=generated)
