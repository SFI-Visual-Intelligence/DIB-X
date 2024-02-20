import torch
from torch.utils.data import DataLoader
from data.echosounder_sampler import SampleEchosounderData128
from data.seal_sampler import SampleSealData128
from data.mnist_sampler import SampleMnistData28

class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"

def return_data(args):
    if args.data == 'seal':
        data = SampleSealData128(args=args)

    elif args.data == 'echograms':
        data = SampleEchosounderData128(args=args)

    elif args.data == 'mnist':
        data = SampleMnistData28()

    else:
        raise Exception('check args.data if it is seal or echograms')

    train_data = data.for_train()
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               drop_last=True,
                                               pin_memory=True,
                                               )
    val_data = data.for_val()
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               drop_last=True,
                                               pin_memory=True,
                                               )

    test_data = data.for_test()
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              drop_last=True,
                                              pin_memory=True,
                                              )

    visual_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              drop_last=True,
                                              pin_memory=True,
                                              )


    if args.data == 'echograms':
        mixed_data = data.for_mixed()
        mixed_loader = torch.utils.data.DataLoader(mixed_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  drop_last=True,
                                                  pin_memory=True,
                                                  )

    data_loader = dict()
    data_loader['x_type'] = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    data_loader['y_type'] = torch.cuda.LongTensor if args.cuda else torch.LongTensor
    data_loader['train'] = train_loader
    data_loader['valid'] = val_loader
    data_loader['test'] = test_loader
    data_loader['visual'] = visual_loader
    if args.data == 'echograms':
        data_loader['mixed'] = mixed_loader

    return data_loader


