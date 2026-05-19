import torch
from .initialize import initialize_megatron

from .global_vars import get_args




def pretrain(
    model_provider,
    train_valid_test_dataset_provider,
    extra_args_provider=None,
    args_defaults={},
):
    
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
    )
    
    args = get_args()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)


    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)


    train()
    
    
    
    


def setup_model_and_optimizer(model_provider):
    args = get_args()
    model = get_model(model_provider)




def get_model(model_provider):
    





def build_train_valid_test_data_iterators(train_valid_test_dataset_provider):
    pass



def train():
    pass


