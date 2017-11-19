from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import tqdm

#from . import module
#from . import utils
#from module import discriminator, generator_unet, generator_resnet, abs_criterion, mae_criterion, sce_criterion
#from utils import ImagePool, load_train_data, save_images, load_test_data
import module
import utils

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size   # 1
        self.image_size = args.fine_size    # 256
        self.input_c_dim = args.input_nc    # 3
        self.output_c_dim = args.output_nc  # 3
        self.L1_lambda = args.L1_lambda     # 10
        self.dataset_dir = args.dataset_dir # horse2zebra

        self.discriminator = module.discriminator
        if args.use_resnet: # True
            self.generator = module.generator_resnet
        else:
            self.generator = module.generator_unet
        if args.use_lsgan:  # True
            self.criterionGAN = module.mae_criterion
        else:
            self.criterionGAN = module.sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,      # 1, 256
                                      args.ngf, args.ndf, args.output_nc,   # 64, 64, 3
                                      args.phase == 'train'))               # train

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = utils.ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim] # batch_size = 1
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        
        # isn't it ones_like -> zeros_like ? :JM
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * module.abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * module.abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * module.abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * module.abs_criterion(self.real_B, self.fake_B_)
            
        ### g_loss      # isn't it ones_like -> zeros_like ? Nope! because it is generator loss!! :JM
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * module.abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * module.abs_criterion(self.real_B, self.fake_B_)

        # feeddict fake_A and fake_B later :JM
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        
        # DB_fake 에서 사용한 discriminator 의 variable들을 그대로 사용하겠다 --> reuse :JM
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        
        ### d_loss
        self.d_loss = self.da_loss + self.db_loss

        # summary
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(args.epoch): # 200
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            print("Epoch: [%2d], time: %4.4f" % (epoch, time.time() - start_time))
            for idx in tqdm(range(0, batch_idxs)): # len(dataA) = 1067
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [utils.load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32) # (1,256,256,6)

                # update Network recursively by G followed by D :JM
                
                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
#                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
#                    epoch, idx, batch_idxs, time.time() - start_time)))

                # I think this should be change, for example, sampling same images :JM
                if np.mod(counter, args.print_freq) == 1:   # print_freq = 5000
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:    # save_freq = 1000
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        bs = 120 # batch size
        step = 6
#        extract = np.arange(0,bs,step)
        fake_A_dir = [dataB[n].split('/')[-1].split('.')[0] for n in range(0,bs,step)]
        fake_B_dir = [dataA[n].split('/')[-1].split('.')[0] for n in range(0,bs,step)]
        if epoch == 0:
            for i in range(20):
                try: 
                    os.makedirs('./{}/fake_A/{}'.format(self.dataset_dir,fake_A_dir[i]))
                    os.makedirs('./{}/fake_B/{}'.format(self.dataset_dir,fake_B_dir[i]))
                except: pass
#        np.random.shuffle(dataA)
#        np.random.shuffle(dataB)
#        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        batch_files = list(zip(dataA[0:bs:step], dataB[0:bs:step]))
        sample_images = [utils.load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32) # (bs/step,256,256,6)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
#        utils.save_images(fake_A, [self.batch_size, 1],
#                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
#        utils.save_images(fake_B, [self.batch_size, 1],
#                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        utils.save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx),fake_A_dir)
        utils.save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx),fake_B_dir)

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [utils.load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            utils.save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
