import numpy as np
import argparse
import os
import torch
import time
from data_loader import get_dataloaders
from jambatalk import JambaTalk
import os, shutil

@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    # Load checkpoint
    checkpoint_path = os.path.join(save_path, f'{epoch}_model.pth')
    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, emotion_id, intensity_id, file_name in test_loader:
        # Move to GPU
        audio = audio.to(device="cuda")
        vertice = vertice.to(device="cuda")
        template = template.to(device="cuda")
        
        # Process emotion and intensity conditioning
        emotion_id = emotion_id.to(device="cuda", dtype=torch.long)
        if emotion_id.ndim > 1:
            emotion_id = emotion_id.squeeze()
        
        intensity_id = intensity_id.to(device="cuda", dtype=torch.float32)
        if intensity_id.ndim == 0:
            intensity_id = intensity_id.view(1, 1)
        elif intensity_id.ndim == 1:
            intensity_id = intensity_id.unsqueeze(-1)
        
        train_subject = "_".join(file_name[0].split("_")[:-1])
        
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            # Pass emotion and intensity to predict
            prediction, lip_features, logits = model.predict(
                audio, template, 
                emotion_id=emotion_id, 
                intensity=intensity_id
            )
            prediction = prediction.squeeze()  # (seq_len, V*3)
            output_file = os.path.join(
                result_path, 
                file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"
            )
            np.save(output_file, prediction.detach().cpu().numpy())
            print(f"Saved: {output_file}")
        else:
            for iter in range(len(train_subjects_list)):  # All training subjects
                condition_subject = train_subjects_list[iter]
                # Pass emotion and intensity to predict
                prediction, lip_features, logits = model.predict(
                    audio, template,
                    emotion_id=emotion_id,
                    intensity=intensity_id
                )
                prediction = prediction.squeeze()  # (seq_len, V*3)
                output_file = os.path.join(
                    result_path,
                    file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"
                )
                np.save(output_file, prediction.detach().cpu().numpy())
                print(f"Saved: {output_file}")

         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='JambaTalk Testing')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="EmoVOCA", help='EmoVOCA')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=512, help='512 for vocaset; 1024 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="sequences", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--landmarks_path", type=str, default="landmarks_sequences", help='path of the ground truth')
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='template_file')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--last_train", type=int, default=0, help='last train') 
    parser.add_argument("--load_path", type=str, default=None, help='path to the trained models')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    parser.add_argument("--emotions", type=str, nargs='+',
                        default=['Smile2', 'Irritated1', 'Sad1', 'Upset', 'Afraid', 'Disgust'], 
                        help='Afraid Disgust Drunk2 ill Irritated1 Moody Pleased Sad1 Smile2 Suspicious Upset')
    parser.add_argument("--intensities", type=str, nargs='+', 
                        default=['1', '2', '3'], 
                        help='intensity levels = [1 2 3]') 
    parser.add_argument("--save_path", type=str, default="save_512_12_10_22_42", help='path of the trained models')
    parser.add_argument("--test_emotion", type=str, default='Afraid', 
                        help='Specific emotion to test (must be in emotions list)')
    parser.add_argument("--test_intensity", type=str, default='2', 
                        help='Specific intensity to test (must be in intensities list)')
    
    args = parser.parse_args() 

    # Validate test emotion/intensity
    if args.test_emotion not in args.emotions:
        print(f"Warning: test_emotion '{args.test_emotion}' not in emotions list: {args.emotions}")
        print(f"Using first emotion: {args.emotions[0]}")
        args.test_emotion = args.emotions[0]
    
    if args.test_intensity not in args.intensities:
        print(f"Warning: test_intensity '{args.test_intensity}' not in intensities list: {args.intensities}")
        print(f"Using first intensity: {args.intensities[0]}")
        args.test_intensity = args.intensities[0]

    print(f"Testing with emotion: {args.test_emotion}, intensity: {args.test_intensity}")
    print(f"Model trained on {len(args.emotions)} emotions: {args.emotions}")
    print(f"Model trained on {len(args.intensities)} intensities: {args.intensities}")

    # Build model
    model = JambaTalk(args)
    print("Model parameters:", count_parameters(model))

    # To CUDA
    assert torch.cuda.is_available(), "CUDA not available"
    model = model.to(args.device)

    # Load data
    dataset = get_dataloaders(args)
    
    # Test
    test(args, model, dataset["test"], epoch=args.max_epoch)
    print(f"\nTesting complete! Results saved to: {os.path.join(args.dataset, args.result_path)}")


if __name__ == "__main__":
    main()