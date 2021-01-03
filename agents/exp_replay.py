import torch
import os
import numpy as np
from importlib import import_module
from .default import NormalNN
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage, AppendName, Permutation
from dataloaders.gan_dataset import *


class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]


class Naive_Rehearsal(NormalNN):

    def __init__(self, agent_config):
        super(Naive_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 1000
        self.task_memory = {}


    def learn_batch(self, train_loader, val_loader=None):
        print('rehersal learn batch')
        # 1.Combine training set
        dataset_list = []
        for storage in self.task_memory.values():
            dataset_list.append(storage)
        dataset_list *= max(len(train_loader.dataset)//self.memory_size,1)  # Let old data: new data = 1:1
        dataset_list.append(train_loader.dataset)
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        new_train_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=train_loader.batch_size,
                                                       shuffle=True,
                                                       num_workers=train_loader.num_workers)

        # 2.Update model as normal
        super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)

        # 3.Randomly decide the images to stay in the memory
        self.task_count += 1
        print('naive_rehersal task count', self.task_count)
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)
        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(train_loader.dataset[ind])


class Naive_Rehearsal_SI(Naive_Rehearsal, SI):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_SI, self).__init__(agent_config)


class Naive_Rehearsal_L2(Naive_Rehearsal, L2):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_L2, self).__init__(agent_config)


class Naive_Rehearsal_EWC(Naive_Rehearsal, EWC):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_EWC, self).__init__(agent_config)
        self.online_reg = True  # Online EWC


class Naive_Rehearsal_MAS(Naive_Rehearsal, MAS):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_MAS, self).__init__(agent_config)

class AGEM(Naive_Rehearsal):
    """
    modified from the code url={https://github.com/facebookresearch/agem}

    @inproceedings{AGEM,
      title={Efficient Lifelong Learning with A-GEM},
      author={Chaudhry, Arslan and Ranzato, Marc’Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
      booktitle={ICLR},
      year={2019}
    }

    @article{chaudhryER_2019,
      title={Continual Learning with Tiny Episodic Memories},
      author={Chaudhry, Arslan and Rohrbach, Marcus and Elhoseiny, Mohamed and Ajanthan, Thalaiyasingam and Dokania, Puneet K and Torr, Philip HS and Ranzato, Marc’Aurelio},
      journal={arXiv preprint arXiv:1902.10486, 2019},
      year={2019}
    }

    @inproceedings{chaudhry2018riemannian,
      title={Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence},
      author={Chaudhry, Arslan and Dokania, Puneet K and Ajanthan, Thalaiyasingam and Torr, Philip HS},
      booktitle={ECCV},
      year={2018}
    }
    """
    def __init__(self, agent_config):
        super(AGEM, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.task_grads = {}
        self.task_mem_cache = {}
        self.gan_add = False
        if agent_config['gan_add']:
            print('use gan method')
            self.gan_add = True
            # current_dir = os.getcwd()
            # self.replicate_pattern = agent_config['replicate_pattern']
            # self.generator_path = os.path.join(current_dir, 'model_file/Generator_cpu_75.pt')
            # self.discriminator_path = os.path.join(current_dir, 'model_file/Discriminator_cpu_75.pt')
            # print('generator:', self.generator_path)
            self.replicate_pattern = agent_config['replicate_pattern']
            self.generator_path = agent_config['generator_path']
            self.discriminator_path = agent_config['discriminator_path']
            print('generator:', self.generator_path)
    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def get_grad(self, current_grad, previous_avg_grad):
        #print(memories_np.shape, gradient_np.shape)
        dotp = (current_grad * previous_avg_grad).sum()
        ref_mag = (previous_avg_grad * previous_avg_grad).sum()
        new_grad = current_grad - ((dotp / ref_mag) *  previous_avg_grad)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    def gan_dataloader(self, dataset_num):
        GAN = CGAN_(self.discriminator_path, self.generator_path)
        label = GAN.generate_label(dataset_num)
        img = GAN.generate_image(label)
        data_pack = {}
        name = str(self.task_count // 2)

        normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))
        transformation = transforms.Compose([
                transforms.Pad(2, fill=0, padding_mode='constant'),
                transforms.ToTensor(),
                normalize,])
        dataset = GAN_MNIST(img, label, transform=transformation)

        # do the permutation and target name 
        remap_class = True
        first_class_ind = (int(name)-1)*dataset.number_classes if remap_class else 0
        # dataset = AppendName(Permutation(dataset, self.replicate_pattern[self.task_count]),
        #     str(self.task_count), first_class_ind=first_class_ind)
        # return  dataset

        # create the dataset in the dict version
        data_pack[name] = AppendName(Permutation(dataset, self.replicate_pattern[int(name)]),
            name, first_class_ind=first_class_ind)
        # create corresbonded dataloader
        gan_dataloader = torch.utils.data.DataLoader(data_pack[name],
                                    batch_size=128, shuffle=True, num_workers=2)

        return gan_dataloader

    def learn_batch(self, train_loader, val_loader=None):

        # 1.Update model as normal
        super(AGEM, self).learn_batch(train_loader, val_loader)
        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1
        print('learn batch task count', self.task_count)
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        print('agem learn_batch')

        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
                storage.reduce(num_sample_per_task)
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot


        if self.gan_add == False:
            # (c) Randomly choose some samples from new task and save them to the memory

            randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
            for ind in randind:  # save it to the memory
                self.task_memory[self.task_count].append(train_loader.dataset[ind])
        
        elif self.gan_add == True:
            # generate as much as the request of data memory
            gan_dataloader = self.gan_dataloader(num_sample_per_task)
            for ind in range(num_sample_per_task):  # save it to the memory
                self.task_memory[self.task_count].append(gan_dataloader.dataset[ind])


        # (d) Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            assert len(mem_loader)==1,'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            # print("mem_task", mem_task)
            # print("self_task", self.task_count)
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}

        # seal into data loader


    def update_model(self, inputs, targets, tasks):

        # compute gradient on previous tasks
        if self.task_count > 0:
            # print('taskcount', self.task_count)
            for t,mem in self.task_memory.items():
                self.zero_grad()
                # feed the data from memory and collect the gradients
                mem_out = self.forward(self.task_mem_cache[t]['data'])
                # if self.task_count > 2:
                    # print('task count', self.task_count)
                    # print(mem_out, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                mem_loss = self.criterion(mem_out, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                mem_loss.backward()
                # Store the grads
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on the current minibatch
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            # reference grad should be average gradient of all previous tasks
            ref_grad_vec = torch.stack(list(self.task_grads.values()))
            ref_grad_vec = torch.sum(ref_grad_vec, dim=0)/ref_grad_vec.shape[0]
            assert current_grad_vec.shape == ref_grad_vec.shape
            dotp = current_grad_vec * ref_grad_vec
            dotp = dotp.sum()
            if (dotp < 0).sum() != 0:
                new_grad = self.get_grad(current_grad_vec, ref_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach(), out



class GEM(Naive_Rehearsal):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.quadprog = import_module('quadprog')
        self.task_mem_cache = {}

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project2cone2(self, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        margin = self.config['reg_coef']
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        #print(memories_np.shape, gradient_np.shape)
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = self.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    def learn_batch(self, train_loader, val_loader=None):

        # 1.Update model as normal
        super(GEM, self).learn_batch(train_loader, val_loader)

        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)
        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(train_loader.dataset[ind])
        # (d) Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            assert len(mem_loader)==1,'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}

    def update_model(self, inputs, targets, tasks):

        # compute gradient on previous tasks
        if self.task_count > 0:
            for t,mem in self.task_memory.items():
                self.zero_grad()
                # feed the data from memory and collect the gradients
                mem_out = self.forward(self.task_mem_cache[t]['data'])
                mem_loss = self.criterion(mem_out, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                mem_loss.backward()
                # Store the grads
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on the current minibatch
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach(), out
