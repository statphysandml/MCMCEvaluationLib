

# site, moving dimension, direction, index of site representation
def get_neighbour_index(n, dim, direction, mu, dimensions, dim_mul, elem_per_site):
    if direction:
        return int((n - n % (dim_mul[dim] * dimensions[dim]) +
                    (n + dim_mul[dim]) % (dim_mul[dim] * dimensions[dim])) * elem_per_site + mu)
    else:
        return int((n - n % (dim_mul[dim] * dimensions[dim]) +
                    (n - dim_mul[dim] + dim_mul[dim] * dimensions[dim]) % (dim_mul[dim] * dimensions[dim])) * elem_per_site + mu)