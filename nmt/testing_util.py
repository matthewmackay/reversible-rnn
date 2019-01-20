import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return 57.2958 * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def create_grad_dict(model):
    grads = {}
    for param in model.parameters():
        grads[param] = param.grad.data.clone().numpy()
    return grads

def compare_grads(grads1, grads2):
    print("COMPARING GRADS")
    for k in grads1:
        flatten1 = grads1[k].flatten()
        flatten2 = grads2[k].flatten()
        print("Angle: " + str(angle_between(flatten1, flatten2)))

def detail_grads(grads1, grads2):
    pass