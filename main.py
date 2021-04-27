import torch
torch.manual_seed(1)
torch.cuda.seed_all()
torch.cuda.manual_seed_all(1)
from tqdm import tqdm
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset import DatasetFromFolder
from model import Generator, Discriminator,  VGGPerceptualLoss
import utils
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='rl_cpy', help='input dataset')
parser.add_argument('--res_dir', required=False, default='1', help='input dataset')
parser.add_argument('--direction', required=False, default='AtoB', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=32, help='train batch size')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--ckpt_s', type=bool, default=True, help='Checkpoint Save')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.00001, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.00001, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--alph', type=float, default=0.1, help='alpha for VGG Perceptual loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--multi_gpu', type=bool, default=False)
params = parser.parse_args()
print(params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories for loading data and saving results
data_dir = 'Data/' + params.dataset + '/'
pth = 'results/rs_' + params.res_dir + '/'
save_dir = 'results/rs_' + params.res_dir + '/' + params.dataset + '_results/'
model_dir = 'results/rs_' + params.res_dir + '/' + params.dataset + '_model/'
csv_dir = 'results/rs_' + params.res_dir + '/' + params.dataset + '_CSV/'
ckpt_dir = 'results/rs_' + params.res_dir + '/' + params.dataset + '_ckpt/'

if not os.path.exists(save_dir):
    os.mkdir(pth)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)


G_ckpt_dir = ckpt_dir + "G_model.pt"
D_ckpt_dir = ckpt_dir + "D_model.pt"




# Data pre-processing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((params.input_size,params.input_size)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Train data
train_data = DatasetFromFolder(data_dir, subfolderA='trainA', subfolderB='trainB', direction=params.direction, transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True)

# Test data
test_data = DatasetFromFolder(data_dir, subfolderA='testA', subfolderB='testB', direction=params.direction, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()

# Models
G = Generator(3, params.ngf, 3)
D = Discriminator(6, params.ndf, 1)

if (params.multi_gpu) and (torch.cuda.device_count() > 1):
    print('Use {} GPUs'.format(torch.cuda.device_count()))
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)

G.to(device)
D.to(device)

# Set the logger
D_log_dir = save_dir + 'D_logs'
G_log_dir = save_dir + 'G_logs'
Img_log_dir = save_dir + 'IMG_logs'

if not os.path.exists(D_log_dir):
    os.mkdir(D_log_dir)
D_logger = SummaryWriter(D_log_dir)

if not os.path.exists(G_log_dir):
    os.mkdir(G_log_dir)
G_logger = SummaryWriter(G_log_dir)

if not os.path.exists(Img_log_dir):
    os.mkdir(Img_log_dir)
Img_logger = SummaryWriter(Img_log_dir)

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
VGG_Loss = VGGPerceptualLoss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2), weight_decay=1e-6)
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2), weight_decay=1e-6)

# Training GAN
D_avg_losses = []
G_avg_losses = []

epochs = 0
step = 0

# Check if Checkpoint exists if yes load them
try:
    f = open(G_ckpt_dir)
    f = open(D_ckpt_dir)
except:
    print("--------Continuing Without Loaded Checkpoint--------")
else:
    checkpoint = torch.load(G_ckpt_dir)
    G.load_state_dict(checkpoint['model_state_dict'])
    G_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    checkpoint = torch.load(D_ckpt_dir)
    D.load_state_dict(checkpoint['model_state_dict'])
    D_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    step = epochs * len(train_data_loader)
    print(f"--------Successfully Loaded Checkpoints till: {str(epochs)}--------")

print("--------Training Started--------")

for epoch in range(epochs, params.num_epochs):
    D_losses = []
    G_losses = []

    # training
    for i, (input, target) in tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False):

        # input & target image data
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = params.lamb * L1_loss(gen_image, y_)

        # VGG Feature Extract
        VGG_L1_Loss = params.alph * VGG_Loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss + VGG_L1_Loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        # ============ TensorBoard logging ============#
        D_logger.add_scalar('D_loss', D_loss.item(), step + 1)
        G_logger.add_scalar('G_loss', G_loss.item(), step + 1)
        
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    print('Epoch [%d/%d], D_loss: %.4f, G_loss: %.4f'
        % (epoch+1, params.num_epochs, D_avg_loss, G_avg_loss))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # Show result for test image
    gen_image = G(Variable(test_input.cuda()))
    gen_image = gen_image.cpu().data
    r = np.random.randint(0,len(test_input))
    Img_logger.add_image("Input", test_input[r], step)
    Img_logger.add_image("Generated", gen_image[r], step)
    Img_logger.add_image("Target", test_target[r], step)
    if epoch % 10 == 0:
        print(f"--------Testing for Epoch No: {str(epoch)}--------")
        utils.test(test_data_loader, G, device, epoch, save_dir, csv_dir)
    
    
    # Save Checkpoints
    if params.ckpt_s:
        utils.save_checkpoint(epoch, G, G_optimizer, G_ckpt_dir)
        utils.save_checkpoint(epoch, D, D_optimizer, D_ckpt_dir)

# Plot average losses
utils.plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Save trained parameters of model
torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')
