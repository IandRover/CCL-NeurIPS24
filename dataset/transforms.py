import torch
import copy

def data_transform(args, data):
    
    if args.task_transform == "same":
        return data, data
    elif args.task_transform == "DropColor":
        data_fw = copy.deepcopy(data)
        for data_idx in range(len(data_fw)):
            channel_id = torch.randint(5,[1])
            if channel_id < 3:
                data_fw[data_idx, channel_id] = 0
        return data_fw, data
    elif args.task_transform == "RandCrop":
        # Randomly crop 32x32 from 96 x 96 image
        data_fw = copy.deepcopy(data)
        for data_idx in range(len(data_fw)):
            if torch.rand([1]) < 0.5:
                h, w = torch.randint(64,[1]), torch.randint(64,[1])
                data_fw[data_idx,:,h:h+32,w:w+32] = 0
        return data_fw, data
    elif args.task_transform == "rot90":
        return data, torch.rot90(data, dims=[2,3])
    elif args.task_transform == "rot180":
        temp = torch.rot90(data, dims=[2,3])
        return data, torch.rot90(temp, dims=[2,3])
    elif args.task_transform == "shiftv5":
        return data, torch.roll(data, 5, dims=[2])
    elif args.task_transform == "shiftv10":
        return data, torch.roll(data, 10, dims=[2])
    elif args.task_transform == "shifth5":
        return data, torch.roll(data, 5, dims=[3])
    elif args.task_transform == "shifth10":
        return data, torch.roll(data, 10, dims=[3])
    elif args.task_transform == "bright":
        return data, 1 - data