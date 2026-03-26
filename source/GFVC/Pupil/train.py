from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from mydataset import myDataset,DatasetRepeater


def train(config, generator, discriminator, checkpoint, log_dir, dataset, my_device, eye='right'):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    # generator = torch.load("generator.pth")
    # discriminator = torch.load("discriminator.pth")
    # optimizer_generator= torch.load("optimizer_generator.pth")
    # optimizer_discriminator=torch.load("optimizer_discriminator.pth")


    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator,
                                      optimizer_generator, optimizer_discriminator)
    else:
        start_epoch = 0

    # start_epoch=1

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    generator_full = GeneratorFullModel(generator, discriminator, train_params,my_device)
    generator_full.to(my_device)
    discriminator_full = DiscriminatorFullModel(generator, discriminator, train_params)
    discriminator_full.to(my_device)

    # if torch.cuda.is_available():
    #     generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    #     discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    cnt_train = 0
    cnt_epoch = start_epoch
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq'],eye=eye) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            loss_list = []
            for x in dataloader:

                source_img = x["source_img"]
                source_img = source_img.to(my_device)
                source_lms = x["source_lms"]
                source_lms = source_lms.to(my_device)
                driving_img = x["driving_img"]
                driving_img = driving_img.to(my_device)
                driving_lms = x["driving_lms"]
                driving_lms = driving_lms.to(my_device)
                losses_generator, generated = generator_full(source_img, source_lms, driving_img, driving_lms)


                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss_list.append(loss)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()


                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(source_img, source_lms, driving_img, driving_lms, generated)

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
                cnt_train = cnt_train + 1
                # my_loss = 0
                # for l in losses:
                #     my_loss += losses[l]
                # loss_list.append(my_loss)
                # print("train:{} ".format(cnt_train)+"loss:{}".format(losses.items()))

            scheduler_generator.step()
            scheduler_discriminator.step()
            # cnt_epoch = cnt_epoch + 1
            # my_loss = sum(loss_list)/len(loss_list)
            # print("------------epoch:{},loss:{}----------".format(cnt_epoch, my_loss))

            # torch.save(generator,"generator.pth")
            # torch.save(discriminator, "discriminator.pth")
            # torch.save(optimizer_generator, "optimizer_generator.pth")
            # torch.save(optimizer_discriminator, "optimizer_discriminator.pth")
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator}, inp=x, out=generated)


