import os
import torch
from torch.utils.data import DataLoader

from options import get_test_options
from datasets import UnalignedDataset
from models import CycleGANModel
from utils import tensor2im, save_image

def main():
    opt = get_test_options()
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.display_id = -1   # no visdom display

    # Dataset
    dataset = UnalignedDataset(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads)
    )

    # Model definition
    model = CycleGANModel(opt)
    
    # Load Weights
    # For testing we only really need generators, but CycleGANModel loads generators automatically
    # We will try loading "latest" by default, or you can specify epoch using --epoch_count if implemented
    model.load_networks('latest')
    
    if opt.eval:
        model.netG_A.eval()
        model.netG_B.eval()

    # Create html directory
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_latest')
    image_dir = os.path.join(web_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    # HTML File Generation
    html_path = os.path.join(web_dir, 'index.html')
    html_file = open(html_path, 'wt')
    html_file.write('<html><body><table border="1">\n<tr><th>Input File</th><th>Input Image</th><th>Output Image</th></tr>\n')

    print(f"Executing Inference. Saving results to: {web_dir}")

    for i, data in enumerate(dataloader):
        if i >= opt.num_test:  # only test our model on opt.num_test images.
            break
            
        model.set_input(data)
        with torch.no_grad():
            if opt.direction == 'AtoB':
                output = model.netG_A(model.real_A)
                real = model.real_A
            else:
                output = model.netG_B(model.real_B)
                real = model.real_B
                
        img_path = model.image_paths[0]
        short_path = os.path.basename(img_path)
        name, ext = os.path.splitext(short_path)
        
        real_img = tensor2im(real)
        fake_img = tensor2im(output)

        real_save_path = os.path.join(image_dir, f"{name}_real{ext}")
        fake_save_path = os.path.join(image_dir, f"{name}_fake{ext}")
        
        save_image(real_img, real_save_path)
        save_image(fake_img, fake_save_path)
        
        print(f"processed {short_path}")
        html_file.write(f'<tr><td>{short_path}</td><td><img src="images/{name}_real{ext}" width="256"></td><td><img src="images/{name}_fake{ext}" width="256"></td></tr>\n')

    html_file.write('</table></body></html>')
    html_file.close()
    print("Inference Complete.")

if __name__ == '__main__':
    main()
