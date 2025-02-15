import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

expert_assignments = (torch.tensor([10,  1, 13,  8, 13,  7, 13,  7,  7, 14,  7,  7,  7, 14, 15, 10, 13,  7,
        13, 13, 13,  7,  7, 13, 13,  7,  7, 14,  7, 10,  1,  6,  3,  2, 13,  7,
        13,  2,  2, 13,  1,  1, 14, 10,  1,  2, 13, 13,  2,  3,  2, 12,  7,  7,
         2,  1, 10, 10,  1, 13,  2,  2,  2,  7,  2,  2,  2, 13,  1,  2,  2,  2,
        10,  1,  7, 13, 13,  7, 13, 13,  7,  7,  1,  7,  2,  2,  2,  1,  7,  8,
         2, 13, 13,  7,  7,  7,  7,  7,  2,  2,  2,  2,  7, 14,  7,  7, 12,  7,
         7,  7,  2,  2,  2,  2,  2, 15,  1,  7, 13, 13,  2, 12,  7,  7,  2,  2,
         2, 15, 15,  8,  1,  1,  1,  2,  7,  7,  7,  1,  2,  2,  8,  2,  2,  8,
         8,  1, 12, 12, 12, 12, 13,  6,  1,  2,  8,  2,  8,  8,  8,  7, 13, 12,
        12, 13,  1,  1,  1,  2,  8,  8,  8,  8,  8,  1,  7,  7,  7,  7,  1,  1,
         7,  6, 14,  8,  8,  8,  8,  1,  7,  7, 13,  7,  1,  1,  7,  1]), torch.tensor([14, 14,  6,  9,  9,  6,  6, 12, 12,  6,  5, 12,  6, 14, 11,  9,  9,  6,
         6,  3, 12, 12, 12, 12,  0,  0, 12, 12,  9,  6,  3,  3,  3, 12, 12, 12,
        12, 12, 12,  0, 12, 12,  9,  6,  3,  3,  5, 12, 12, 12, 12, 12, 12, 12,
        12, 12,  6,  5,  2,  3, 12, 12, 12, 12, 12,  0, 12, 12, 12, 12,  3,  2,
        12, 12, 12, 12,  6, 12, 12,  0, 12, 12, 12, 12,  5,  3, 12, 12, 12, 12,
        12,  5, 12,  0, 12, 12, 12, 12,  1,  2, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12,  1,  5,  5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        15,  1,  7, 12,  7,  5, 12, 12, 12, 12, 12, 12, 12,  7, 15, 15,  7,  7,
         2, 12,  7, 14,  7,  7,  7,  7, 12,  7, 15, 15,  1,  7,  2, 12, 12, 12,
         1,  1,  5,  1,  1,  1,  9, 15, 15,  1,  1, 12, 12, 12,  7, 12,  5,  1,
         1, 15, 14, 14, 15,  9, 15,  1,  1, 12,  7,  5,  7, 15,  7,  7]
       ), torch.tensor([15, 15, 11, 15, 10, 10, 10, 10, 10, 10, 14,  4, 10, 10,  9, 10, 15, 15,
        14, 10, 10, 10, 10, 10,  4,  4, 10, 10, 14, 15, 15, 15, 15, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 15, 14, 15, 15, 15, 10, 10, 10, 10, 10, 10,  7,
         7,  7, 15, 15, 15,  4,  4, 10, 10, 10, 14, 10, 10,  7, 10, 10,  4,  4,
         5, 10, 10, 10, 10, 10, 10, 10,  7,  7,  7, 10, 15, 10, 10,  2,  2,  6,
         6, 10, 10, 10, 10, 10, 10, 10, 15, 10, 10,  2,  2,  2,  2,  2,  2,  2,
         2, 10,  2,  1, 15, 10, 10,  2, 10,  2,  2,  2,  2,  2, 10, 10, 10,  1,
        15, 10, 10,  2, 10,  2,  2,  2,  2,  2, 10, 10, 10, 10,  7, 14, 10,  2,
        10,  2,  2,  2, 10,  2, 14, 10, 10,  1,  7, 10, 10,  2,  2,  2, 10,  2,
         2,  2, 10, 10, 10,  1,  7,  1,  1,  1,  3, 10, 14, 10, 10,  2, 10, 10,
         1,  3,  5,  1,  4,  1,  3,  4, 15, 15, 15, 15,  7,  1,  1,  3]
       ), torch.tensor([ 5,  5, 14, 13, 13, 13, 13,  5, 13,  3, 13, 14,  5, 14, 13, 13, 13, 13,
        13, 13, 13, 13,  5,  5,  5,  5, 13,  1,  3, 13, 13, 13, 13, 14, 13, 13,
        13, 13, 13, 13, 13,  1,  3, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 10, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 10,  3, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14,  3, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13,  1, 10, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13,  1, 15,  5,  1,  1, 13, 13, 13, 13, 13, 13, 11,  5, 14,  5,
        15, 15, 15, 15,  1, 13, 13, 14, 14,  1,  5,  0,  5,  5, 11, 15, 15, 11,
         0,  1, 10,  2,  2,  2,  2,  2,  2, 10, 15, 15, 15, 11, 15,  1,  2,  2,
         2,  2,  2,  2,  2,  2, 15,  9, 11, 14,  0, 15,  2,  2,  2,  2,  2,  2,
         6,  2, 10, 10, 15,  9,  0, 14,  2,  2,  6,  2,  2,  2, 14,  2]))



def create_expert_heatmap_per_layer_torch(expert_assignments, patch_size, num_experts, layer_index, batch_size):
    """
    Create a heatmap for a specific layer where each patch is colored according to the assigned expert using PyTorch.
    
    :param image: Original image as a torch tensor of shape (H, W, C)
    :param patch_size: Tuple (patch_height, patch_width)
    :param expert_assignments: 1D torch tensor of shape (num_patches), containing expert IDs (0 to num_experts - 1)
    :param num_experts: The total number of experts
    :param layer_index: Index of the current layer
    :return: Heatmap image as a torch tensor of shape (H, W, C)
    """
    
    fig, axes = plt.subplots(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), figsize=(8, 8))

    #expert_assignments_batches = expert_assignments[layer_index] # tuple (batch0, batch1, ..., batch_size-1)
    expert_assignments_batches = expert_assignments

    for image_idx, ax in enumerate(axes.flat):
    
        #image = batch[image_idx]
        expert_assignments = expert_assignments_batches[image_idx]
        H, W, C = (224, 224, 3)
        patch_height, patch_width = patch_size, patch_size
        #num_patches_y = H // patch_height
        #num_patches_x = W // patch_width
        num_patches_y = 14
        num_patches_x = 14
        
        # Initialize heatmap with zeros
        #heatmap = torch.zeros((H, W), dtype=torch.float32)
        heatmap = torch.zeros((56,56), dtype = torch.float32)
        # Generate distinct colors for each expert using matplotlib's colormap
        colors = plt.cm.get_cmap('viridis', num_experts)
        
        # Iterate over all patches using the flattened assignment vector
        patch_idx = 0

        for i in range(num_patches_y):
            for j in range(num_patches_x):
                #breakpoint()
                expert_id = expert_assignments[patch_idx].item()  # Get the expert ID for the current patch
                color = torch.tensor(colors(expert_id))  # Get RGB values as a torch tensor
                
                # Assign this color to the corresponding patch in the heatmap
                heatmap[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width] = color
                
                patch_idx += 1
        
        ax.imshow(heatmap.numpy())
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('heatmaps.png')
    plt.show()
    return

#create_expert_heatmap_per_layer_torch(expert_assignments=expert_assignments, patch_size=4, num_experts=16, layer_index=17, batch_size=4)

# Generate a random 196-dimensional tensor with values from 0 to 16
#tensor196 = torch.randint(0, 17, (196,))

def heatmap2(expert_assignments):
# Reshape the tensor to a 14x14 grid
    tensor14x14 = expert_assignments.view(14, 14)

    # Convert the tensor to a NumPy array for plotting
    img = tensor14x14.numpy()

    # Create a colormap with 17 distinct colors for labels 0 to 16
    colors = plt.cm.get_cmap('tab20', 17)
    cmap = ListedColormap(colors.colors)

    # Plot the image
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap=cmap, interpolation='nearest')
    #plt.colorbar(ticks=range(17))
    plt.title('14x14 Image Reconstructed from 196-Dimensional Tensor')
    plt.axis('off')
    plt.savefig('test.png')
    plt.show()

    # Verify the output reconstructs a 14x14 image
    assert img.shape == (14, 14), "The reshaped image does not have dimensions 14x14."

    return

#heatmap2(expert_assignments[0])


def plot_label_tensors(tensor_tuple, name):
    """
    Plots a grid of images from a tuple of 196-dimensional tensors.
    Each tensor is reshaped into a 14x14 image, and images are arranged in a grid.
    """
    num_tensors = len(tensor_tuple)
    grid_size = int(np.sqrt(num_tensors))
    assert grid_size ** 2 == num_tensors, "Number of tensors must be a perfect square."
    
    # Create a colormap with 16 distinct colors for labels 0 to 15
    colors = plt.cm.get_cmap('tab20', 16)
    cmap = ListedColormap(colors.colors)
    
    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4))
    
    # Handle the case when there's only one tensor
    if num_tensors == 1:
        axes = np.array([[axes]])
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Loop over tensors and plot them
    for idx, tensor in enumerate(tensor_tuple):
        # Reshape tensor to 14x14
        img = tensor[:,0].view(14, 14).cpu().numpy() # just take top expert selection
        # Get the corresponding axis
        ax = axes[idx]
        # Plot the image
        im = ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(f"Image {idx+1}")
        ax.axis('off')
    
    # Hide any unused subplots
    for ax in axes[num_tensors:]:
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    # Add a colorbar
    fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    #fig.colorbar(im, cax=cbar_ax, ticks=range(17))
    plt.savefig(name)
    plt.show()

def plot_mixed_label_tensors(label_tensor_tuple, prob_tensor_tuple, name):
    """
    Plots a grid of images from tuples of label tensors and probability tensors.
    Each image is constructed by mixing the colors of two labels for each patch,
    weighted by their associated probabilities.

    Parameters:
    - label_tensor_tuple: tuple of tensors, each of shape (196, 2), containing label indices.
    - prob_tensor_tuple: tuple of tensors, each of shape (196, 2), containing probabilities.

    Assumptions:
    - The number of images (tensors in the tuples) is a perfect square.
    - Labels are integers from 0 to 16.
    - Probabilities for each patch sum to 1.
    """
    num_tensors = len(label_tensor_tuple)
    grid_size = int(np.sqrt(num_tensors))
    assert grid_size ** 2 == num_tensors, "Number of tensors must be a perfect square."
    assert len(label_tensor_tuple) == len(prob_tensor_tuple), "Label and probability tuples must be of the same length."

    # Create a colormap with 16 distinct colors for labels 0 to 15
    colors = plt.cm.get_cmap('tab20', 17)
    cmap = ListedColormap(colors.colors)

    # Get the RGBA colors from the colormap
    label_colors = cmap(np.arange(16))  # Shape: (16, 4)

    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*4, grid_size*4))

    # Handle the case when there's only one tensor
    if num_tensors == 1:
        axes = np.array([[axes]])

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Loop over tensors and plot them
    for idx, (label_tensor, prob_tensor) in enumerate(zip(label_tensor_tuple, prob_tensor_tuple)):
        # Convert tensors to NumPy arrays
        labels_np = label_tensor.cpu().numpy()  # Shape: (196, 2)
        
        prob_tensor = torch.softmax(prob_tensor, axis = 1)
        probs_np = prob_tensor.cpu().numpy()    # Shape: (196, 2)

        # Ensure probabilities sum to 1 for each patch
        probs_sum = probs_np.sum(axis=1)
        #print(probs_sum[0])
        assert np.allclose(probs_sum, 1), f"Probabilities do not sum to 1 for image {idx+1}."

        # Get colors for each label
        color_indices1 = labels_np[:, 0].astype(int)
        color_indices2 = labels_np[:, 1].astype(int)
        colors1 = label_colors[color_indices1]  # Shape: (196, 4)
        colors2 = label_colors[color_indices2]  # Shape: (196, 4)

        # Mix colors weighted by probabilities
        probs1 = probs_np[:, 0][:, np.newaxis]  # Shape: (196, 1)
        probs2 = probs_np[:, 1][:, np.newaxis]  # Shape: (196, 1)
        mixed_colors = probs1 * colors1 + probs2 * colors2  # Shape: (196, 4)

        # Reshape to 14x14x4 for plotting
        img = mixed_colors.reshape(14, 14, 4)

        # Get the corresponding axis
        ax = axes[idx]
        # Plot the image
        im = ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Image {idx+1}")
        ax.axis('off')

    # Hide any unused subplots
    for ax in axes[num_tensors:]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    # Add a colorbar
    fig.subplots_adjust(right=0.85)
    #cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, ticks=range(17))
    #cbar.set_label('Labels')
    plt.savefig(name)
    plt.show()

# Call the function to plot the tensors
#plot_label_tensors(expert_assignments)



#breakpoint()