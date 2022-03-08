

# site, moving dimension, direction, index of site representation
def get_neighbour_index(n, dim, direction, mu, dimensions, dim_mul, elem_per_site=1):
    if direction:
        return int((n - n % (dim_mul[dim] * dimensions[dim]) +
                    (n + dim_mul[dim]) % (dim_mul[dim] * dimensions[dim])) * elem_per_site + mu)
    else:
        return int((n - n % (dim_mul[dim] * dimensions[dim]) +
                    (n - dim_mul[dim] + dim_mul[dim] * dimensions[dim]) % (dim_mul[dim] * dimensions[dim])) * elem_per_site + mu)


def get_neighbour_index_in_line(n, n_next, dim, direction, mu, dimensions, dim_mul, elem_per_site=1):
    n = get_neighbour_index(n=n, dim=dim, direction=direction, mu=mu, dimensions=dimensions, dim_mul=dim_mul,
                            elem_per_site=elem_per_site)
    if n_next == 1:
        return n
    else:
        return get_neighbour_index_in_line(n=n, n_next=n_next-1, dim=dim, direction=direction, mu=mu, dimensions=dimensions, dim_mul=dim_mul, elem_per_site=elem_per_site)
