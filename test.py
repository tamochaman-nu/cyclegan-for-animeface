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
    # We will try loading "latest" by default, or you can specify epoch using --load_epoch
    model.load_networks(opt.load_epoch)
    
    if opt.eval:
        model.netG_A.eval()
        model.netG_B.eval()

    # Create html directory
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_latest')
    dir_AtoB = os.path.join(web_dir, 'AtoB')
    dir_BtoA = os.path.join(web_dir, 'BtoA')
    
    for d in [dir_AtoB, dir_BtoA]:
        os.makedirs(os.path.join(d, 'input'), exist_ok=True)
        os.makedirs(os.path.join(d, 'fake'), exist_ok=True)
        os.makedirs(os.path.join(d, 'restore'), exist_ok=True)
    
    # HTML File Generation
    html_path = os.path.join(web_dir, 'index.html')
    html_file = open(html_path, 'wt')
    html_file.write('<html><head><style>td { text-align: center; font-weight: bold; }</style></head><body>')
    html_file.write('<h2>A to B (e.g. Real -> Anime)</h2><table border="1">\n<tr><th>Input File</th><th>Input Image (A)</th><th>Fake Image (B)</th><th>Restore Image (A)</th></tr>\n')
    
    html_AtoB_rows = []
    html_BtoA_rows = []

    print(f"Executing Inference. Saving results to: {web_dir}")

    for i, data in enumerate(dataloader):
        if i >= opt.num_test:  # only test our model on opt.num_test images.
            break
            
        model.set_input(data)
        with torch.no_grad():
            model.forward()
                
        # Fixed domain concepts: A is always real (Input A), B is always anime (Input B)
        img_A = model.real_A
        fake_B = model.fake_B
        rec_A = model.rec_A
        path_A = data['A_paths'][0]
        
        img_B = model.real_B
        fake_A = model.fake_A
        rec_B = model.rec_B
        path_B = data['B_paths'][0]
            
        name_A, ext_A = os.path.splitext(os.path.basename(path_A))
        name_B, ext_B = os.path.splitext(os.path.basename(path_B))
        
        # Save AtoB results
        save_image(tensor2im(img_A), os.path.join(dir_AtoB, 'input', f"{name_A}{ext_A}"))
        save_image(tensor2im(fake_B), os.path.join(dir_AtoB, 'fake', f"{name_A}{ext_A}"))
        save_image(tensor2im(rec_A), os.path.join(dir_AtoB, 'restore', f"{name_A}{ext_A}"))
        
        # Save BtoA results
        save_image(tensor2im(img_B), os.path.join(dir_BtoA, 'input', f"{name_B}{ext_B}"))
        save_image(tensor2im(fake_A), os.path.join(dir_BtoA, 'fake', f"{name_B}{ext_B}"))
        save_image(tensor2im(rec_B), os.path.join(dir_BtoA, 'restore', f"{name_B}{ext_B}"))
        
        print(f"processed {name_A}{ext_A} (A) and {name_B}{ext_B} (B)")
        html_AtoB_rows.append(f'<tr><td>{name_A}{ext_A}</td><td><img src="AtoB/input/{name_A}{ext_A}" width="256"></td><td><img src="AtoB/fake/{name_A}{ext_A}" width="256"></td><td><img src="AtoB/restore/{name_A}{ext_A}" width="256"></td></tr>\n')
        html_BtoA_rows.append(f'<tr><td>{name_B}{ext_B}</td><td><img src="BtoA/input/{name_B}{ext_B}" width="256"></td><td><img src="BtoA/fake/{name_B}{ext_B}" width="256"></td><td><img src="BtoA/restore/{name_B}{ext_B}" width="256"></td></tr>\n')

    # Write HTML contents
    for row in html_AtoB_rows:
        html_file.write(row)
    html_file.write('</table><hr/>')
    
    html_file.write('<h2>B to A (e.g. Anime -> Real)</h2><table border="1">\n<tr><th>Input File</th><th>Input Image (B)</th><th>Fake Image (A)</th><th>Restore Image (B)</th></tr>\n')
    for row in html_BtoA_rows:
        html_file.write(row)
    html_file.write('</table></body></html>')
    html_file.close()
    print("Inference Complete.")

if __name__ == '__main__':
    main()
