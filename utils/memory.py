import numpy as np
import torch
import faiss

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature, gpu):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.labels = torch.LongTensor(self.n)
        self.labels_str = []
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes
        self.gpu = gpu

        self.ngpus = faiss.get_num_gpus()

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        
        #index = faiss.index_cpu_to_all_gpus(index)
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.gpu, index)
        
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included

        # evaluate 
        if calculate_accuracy:
            labels = self.labels.cpu()
            neighbor_labels = np.take(labels, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_labels = np.repeat(labels.reshape(-1,1), topk, axis=1)
            
            #accuracy = np.mean((neighbor_labels == anchor_labels).numpy())
            corrects = 0
            for i, anchor in enumerate(anchor_labels):
                if anchor in neighbor_labels[i]:
                    corrects += 1
            
            return indices, corrects/len(anchor_labels)
        
        else:
            return indices
        
        return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, labels, labels_str):
        # Batch size
        b = features.size(0)
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.labels[self.ptr:self.ptr+b].copy_(labels.detach())
        self.labels_str.append(labels_str)
        self.ptr += b

    def get_memory(self):
        return self.features.cpu().numpy(), self.labels.cpu().numpy(), self.labels_str

    def to(self, device):
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:{}'.format(self.gpu))

    # Custom wrapper for using various gpus (but not all)
    def my_index_cpu_to_gpu_multiple(resources, index, co=None, gpu_nos=None):
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if gpu_nos is None: 
            gpu_nos = range(len(resources))
        for i, res in zip(gpu_nos, resources):
            vdev.push_back(i)
            vres.push_back(res)
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        index.referenced_objects = resources
        return index


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, forward_pass='default'):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        inputs = batch['clip'].cuda(non_blocking=True)
        labels = batch['label']
        labels_str = batch['label_str']
        output = model(inputs, forward_pass=forward_pass)
        
        # Squeeze any extra dimension
        memory_bank.update(output.reshape(-1, output.shape[1]), labels, labels_str)
        
        #if i % 100 == 0:
        #    print('Fill Memory Bank [%d/%d]' %(i, len(loader)))

import faiss
def mine_nearest_neighbors(features, labels, topk, calculate_accuracy=False):
    features = features.cpu().numpy()
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    index = faiss.index_cpu_to_all_gpus(index)
    #index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    
    index.add(features)
    distances, indices = index.search(features, topk+1) # Sample itself is included

    # evaluate 
    if calculate_accuracy:
        neighbor_labels = np.take(labels, indices[:,1:], axis=0) # Exclude sample itself for eval
        anchor_labels = np.repeat(labels.reshape(-1,1), topk, axis=1)
        
        #accuracy = np.mean((neighbor_labels == anchor_labels).numpy())
        corrects = 0
        for i, anchor in enumerate(anchor_labels):
            if anchor in neighbor_labels[i]:
                corrects += 1
        
        return indices, corrects/len(anchor_labels)
    
    else:
        return indices