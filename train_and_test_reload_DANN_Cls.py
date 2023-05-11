import time
import torch

from helpers import list_of_distances, make_one_hot



def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_aligned = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific

    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label, _) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, v_features, f_preds = model(input)

            # compute loss
            # cross_entropy = torch.nn.functional.cross_entropy(output, target)
            cross_entropy = torch.nn.functional.cross_entropy(output, f_preds.argmax(axis=1))

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_aligned += (predicted == f_preds.argmax(axis=1)).sum().item()
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            if class_specific:
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    log('\talign accu: \t\t{0}%'.format(n_aligned / n_examples * 100))

    return n_correct / n_examples





def _train(model, domain_discri, domain_adv, source_loader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, target_loader=None, num_iters=0):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    # print(domain_discri)

    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_aligned = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    # for i, (image, label) in enumerate(source_loader):
    # print('num_iters: ', num_iters)
    for i in range(num_iters):
        image, label = next(source_loader)[:2]
        input = image.cuda()
        target = label.cuda()

        image_t, = next(target_loader)[:1]
        input_t = image_t.cuda()
        
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            # output, min_distances, _, _ = model(input)

            x = torch.cat((input, input_t), dim=0)
            #log('\tsource samples: \t\t{0}'.format(input))
            #log('\ttarget samples: \t\t{0}'.format(input_t))

            output_full, min_distances_full, v_features_full, f_preds_full = model(x)
            output, output_t = output_full.chunk(2, dim=0)

#            log('\tsource outputs: \t\t{0}'.format(output))
 #           log('\ttarget outputs: \t\t{0}'.format(output_t))


            min_distances, min_distances_t = min_distances_full.chunk(2, dim=0)
            v_features, v_features_t = v_features_full.chunk(2, dim=0)
            f_preds, f_preds_t = f_preds_full.chunk(2, dim=0)


            soft_output_t = torch.nn.functional.softmax(output_t, dim = 1)
            soft_preds_t = torch.nn.functional.softmax(f_preds_t, dim = 1)
            pred_align_loss = torch.nn.functional.l1_loss(soft_output_t, soft_preds_t)
            # compute loss
            # cross_entropy = torch.nn.functional.cross_entropy(output, target)
            cross_entropy = torch.nn.functional.cross_entropy(output_full, f_preds_full.argmax(axis=1))

            # print('proto entropy: ', cross_entropy.item(), 'feature entropy:', feature_entropy_loss.item(), 'transfer_loss:', transfer_loss.item(), 'alignment_loss: ', pred_align_loss.item())
            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                min_distance_t, _ = torch.min(min_distances_t, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_aligned += (predicted == f_preds.argmax(axis=1)).sum().item()
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            if class_specific:
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            
            loss = loss + coefs['align'] * pred_align_loss# +  coefs['fts_ent'] * feature_entropy_loss + coefs['discri'] * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        del input
        del target
        # del output
        del predicted
        #del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
        
    # log('\tsource distance: \t\t{0}'.format(min_distances))
    # log('\ttarget distance: \t\t{0}'.format(min_distances_t))    

    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
            
    # log('\tsource distances: \t\t{0}'.format(min_distance))
    # log('\ttarget distances: \t\t{0}'.format(min_distance_t))
            
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    log('\tPrototype target logits: \t\t{0}'.format(soft_output_t))
    log('\tPrototype target preds: \t\t{0}'.format(torch.argmax(soft_output_t, dim=1)))
    
    log('\tDANN target logits: \t\t{0}'.format(soft_preds_t))
    log('\tDANN target preds: \t\t{0}'.format(torch.argmax(soft_preds_t, dim=1)))

    log('\tweighted pred_align_loss: \t{0}'.format(coefs['align'] * pred_align_loss))
    # log('\tpred_align_loss: \t{0}'.format(pred_align_loss))

    log('\talign accu: \t\t{0}%'.format(n_aligned / n_examples * 100))
    # log('\tDomain Acc: \t\t{0}'.format(domain_acc))
    return n_correct / n_examples






def train(model, domain_discri, domain_adv, dataloader, optimizer, class_specific=False, coefs=None, log=print, target_loader=None, num_iters=0):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train(model=model, domain_discri=domain_discri, domain_adv=domain_adv, source_loader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, target_loader=target_loader, num_iters=num_iters)


def test(model, dataloader, class_specific=False, log=print, target_loader=None):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


# def train_last_only(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
def train_last_only(model, domain_discri, domain_adv, dataloader, optimizer, class_specific=False, coefs=None, log=print, target_loader=None, num_iters=0):

    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    # return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
    #                       class_specific=class_specific, coefs=coefs, log=log)
    return _train(model=model, domain_discri=domain_discri, domain_adv=domain_adv, source_loader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, target_loader=target_loader, num_iters=num_iters)

# def last_only(model, log=print):
#     for p in model.module.features.parameters():
#         p.requires_grad = False
#     for p in model.module.add_on_layers.parameters():
#         p.requires_grad = False
#     model.module.prototype_vectors.requires_grad = False
#     for p in model.module.last_layer.parameters():
#         p.requires_grad = True
    
#     log('\tlast layer')

def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    for p in model.module.bottleneck.parameters():
        p.requires_grad = False    

    for p in model.module.fc.parameters():
        p.requires_grad = False    
    log('\tlast layer')


# def warm_only(model, log=print):
#     for p in model.module.features.parameters():
#         p.requires_grad = False
#     for p in model.module.add_on_layers.parameters():
#         p.requires_grad = True
#     model.module.prototype_vectors.requires_grad = True
#     for p in model.module.last_layer.parameters():
#         p.requires_grad = True
    
#     log('\twarm')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True


    for p in model.module.bottleneck.parameters():
        p.requires_grad = False    

    for p in model.module.fc.parameters():
        p.requires_grad = False        
    log('\twarm')


# def joint(model, log=print):
#     for p in model.module.features.parameters():
#         p.requires_grad = True
#     for p in model.module.add_on_layers.parameters():
#         p.requires_grad = True
#     model.module.prototype_vectors.requires_grad = True
#     for p in model.module.last_layer.parameters():
#         p.requires_grad = True
    
#     log('\tjoint')

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True


    for p in model.module.bottleneck.parameters():
        p.requires_grad = False
    for p in model.module.fc.parameters():
        p.requires_grad = False

    log('\tjoint')

