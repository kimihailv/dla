from tqdm import tqdm
from utils import *


def train_epoch(train_loader, epoch_num,
                generator, msd, mpd,
                fm_loss, adversarial_loss,
                mel_loss, g_optimizer, d_optimizer,
                featurizer, device, logger):
    running_g_loss = 0
    running_d_loss = 0
    num_samples = 0
    generator.train()
    msd.train()
    mpd.train()

    bar = tqdm(train_loader)
    for wavs in bar:
        wavs = wavs.to(device)
        spec = featurizer(wavs.squeeze(1))
        wavs_pred = generator(spec)

        set_requires_grad(mpd, True)
        set_requires_grad(msd, True)
        d_optimizer.zero_grad()
        gen_logits, _ = mpd(wavs_pred.detach())
        real_logits, _ = mpd(wavs)

        mpd_loss = adversarial_loss.calc_discriminator_loss(gen_logits, real_logits)

        gen_logits, _ = msd(wavs_pred.detach())
        real_logits, _ = msd(wavs)

        msd_loss = adversarial_loss.calc_discriminator_loss(gen_logits, real_logits)

        d_loss = mpd_loss + msd_loss

        d_loss.backward()
        d_optimizer.step()

        set_requires_grad(mpd, False)
        set_requires_grad(msd, False)
        g_optimizer.zero_grad()
        mpd_gen_logits, gen_features = mpd(wavs_pred)
        mpd_real_logits, real_features = mpd(wavs)

        mpd_loss_fm = fm_loss(gen_features, real_features)

        msd_gen_logits, gen_features = msd(wavs_pred)
        msd_real_logits, real_features = msd(wavs)

        msd_loss_fm = fm_loss(gen_features, real_features)

        adv_loss = adversarial_loss.calc_generator_loss(mpd_gen_logits)
        adv_loss += adversarial_loss.calc_generator_loss(msd_gen_logits)

        spec_pred = featurizer(wavs_pred.squeeze(1))
        rec_loss = mel_loss(spec_pred, spec)

        g_loss = adv_loss + 45 * (mpd_loss_fm + msd_loss_fm) + 2 * rec_loss
        g_loss.backward()
        g_optimizer.step()

        logger.log({'train_g_loss': g_loss.item(),
                    'train_d_loss': d_loss.item(),
                    'epoch': epoch_num,
                    'iter': num_samples})

        num_samples += 1
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

        bar.set_postfix({
            'Epoch': epoch_num,
            'G Loss': running_g_loss / num_samples,
            'D Loss': running_d_loss / num_samples
        })

    return running_g_loss / num_samples, running_d_loss / num_samples


@torch.no_grad()
def validate(loader, epoch_num, generator, msd, mpd,
             fm_loss, adversarial_loss,
             mel_loss, featurizer, device, logger):
    running_g_loss = 0
    running_d_loss = 0
    num_samples = 0

    generator.eval()
    msd.eval()
    mpd.eval()

    to_log = 5

    bar = tqdm(loader)
    for wavs in bar:
        wavs = wavs.to(device)
        spec = featurizer(wavs.squeeze(1))
        wavs_pred = generator(spec)

        gen_logits, _ = mpd(wavs_pred.detach())
        real_logits, _ = mpd(wavs)

        mpd_loss = adversarial_loss.calc_discriminator_loss(gen_logits, real_logits)

        gen_logits, _ = msd(wavs_pred.detach())
        real_logits, _ = msd(wavs)

        msd_loss = adversarial_loss.calc_discriminator_loss(gen_logits, real_logits)

        d_loss = mpd_loss + msd_loss

        mpd_gen_logits, gen_features = mpd(wavs_pred)
        mpd_real_logits, real_features = mpd(wavs)

        mpd_loss_fm = fm_loss(gen_features, real_features)

        msd_gen_logits, gen_features = msd(wavs_pred)
        msd_real_logits, real_features = msd(wavs)

        msd_loss_fm = fm_loss(gen_features, real_features)

        adv_loss = adversarial_loss.calc_generator_loss(mpd_gen_logits)
        adv_loss += adversarial_loss.calc_generator_loss(msd_gen_logits)

        spec_pred = featurizer(wavs_pred.squeeze(1))
        rec_loss = mel_loss(spec_pred, spec)

        g_loss = adv_loss + 45 * (mpd_loss_fm + msd_loss_fm) + 2 * rec_loss

        num_samples += 1
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

        if to_log > 0:
            sample_idx = torch.randint(0, wavs.size(0), size=(1,)).item()
            logger.add_row(epoch_num,
                           wavs_pred[sample_idx].squeeze().cpu().numpy(),
                           wavs[sample_idx].squeeze().cpu().numpy(),
                           spec_pred[sample_idx].cpu().numpy(),
                           spec[sample_idx].cpu().numpy(),
                           'val')

            to_log -= 1

    logger.push_table('val')

    return running_g_loss / num_samples, running_d_loss / num_samples


def train(train_loader, val_loader,
          n_epochs, generator, msd, mpd,
          fm_loss, adversarial_loss,
          mel_loss, g_optimizer, d_optimizer,
          g_scheduler, d_scheduler, featurizer,
          device, logger, save_every, val_every,
          save_dir):
    for epoch in range(n_epochs):
        train_g_loss, train_d_loss = train_epoch(train_loader, epoch,
                                                 generator, msd, mpd,
                                                 fm_loss, adversarial_loss,
                                                 mel_loss, g_optimizer, d_optimizer,
                                                 featurizer, device, logger)

        logger.log({'train_epoch_g_loss': train_g_loss,
                    'train_epoch_d_loss': train_d_loss,
                    'epoch': epoch})

        g_scheduler.step()
        d_scheduler.step()

        if epoch % val_every == 0:
            val_g_loss, val_d_loss = validate(val_loader, epoch, generator,
                                              msd, mpd, fm_loss, adversarial_loss,
                                              mel_loss, featurizer, device, logger)

            logger.log({'val_epoch_g_loss': val_g_loss,
                        'val_epoch_d_loss': val_d_loss,
                        'epoch': epoch})

        if epoch % save_every == 0:
            state = {
                'generator': generator.state_dict(),
                'msd': msd.state_dict(),
                'mpd': mpd.state_dict(),
                'g_optmizer': g_optimizer.state_dict(),
                'd_optmizer': d_optimizer.state_dict(),
                'g_scheduler': g_scheduler.state_dict(),
                'd_scheduler': d_scheduler.state_dict(),
            }

            torch.save(state, f'{save_dir}/ckp_{epoch}.pt')

    state = {
        'generator': generator.state_dict(),
        'msd': msd.state_dict(),
        'mpd': mpd.state_dict(),
        'g_optmizer': g_optimizer.state_dict(),
        'd_optmizer': d_optimizer.state_dict(),
        'g_scheduler': g_scheduler.state_dict(),
        'd_scheduler': d_scheduler.state_dict(),
    }

    torch.save(state, f'{save_dir}/final.pt')
