import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def to_features(X):
    return X.view(X.size(0), -1)


def image_search(queries, database, reducer_callable, n_neighbors=10, p=2, metric="distance"):
    query_embed = reducer_callable(X=queries)
    database_embed = reducer_callable(X=database)
    if metric == "distance":
        dist = torch.cdist(query_embed, database_embed, p=p)
    elif metric == "cosine":
        dist = 1 - torch.stack([F.cosine_similarity(query_embed[i], database_embed) for i in range(len(query_embed))])
    else:
        raise ValueError(f"metric {metric} is not supported")
    closest_dist, closest_idx = torch.topk(dist, k=n_neighbors, dim=1, largest=False)
    return closest_dist, closest_idx


def compute_img_search_quality(queries, database, query_y, database_y, closest_idx, n_neighbors=5):
    # for each query get the k nearest neighbors
    closest_idx = closest_idx.cpu()
    neighbor_idx = closest_idx[:, :n_neighbors]
    # compute the accuracy which is defined the fraction of database_y of neighbor_idx match with query_y
    neighbor_y = database_y[neighbor_idx]
    accuracy = (neighbor_y == query_y.unsqueeze(1)).sum(dim=1) / n_neighbors
    # accuracy per sample
    return accuracy

def viz_image_search(queries, database, closest_idx, closest_dist, num_images=10, n_neighors=10):
    x_idx = np.random.randint(0, len(queries), num_images)
    x = queries[x_idx]
    closest_idx = closest_idx[x_idx]

    fig, axes = plt.subplots(nrows=num_images, ncols=n_neighbors + 1, figsize=( 10, 12));
    for i, image in enumerate(x):
        axes[i, 0].imshow(image.cpu().squeeze(), cmap="gray");
        axes[i, 0].axis("off");
        for j in range(10):
            axes[i, j+1].imshow(database[closest_idx[i, j]].cpu().squeeze(), cmap="gray");
            axes[i, j+1].title.set_text(f"{closest_dist[i, j]:.2f}");
            axes[i, j+1].axis("off");
    
    plt.show();