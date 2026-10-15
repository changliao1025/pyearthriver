from rtree.index import Index as RTreeindex


def find_vertex_in_list(aVertex_in, pVertex_in, dThreshold_in=1.0e-6):
    """[find the index of a vertex in a list]

    Args:
        aVertex_in ([type]): [description]
        pVertex_in ([type]): [description]

    Returns:
        [type]: [description]
    """

    iFlag_exist = 0
    lIndex = -1
    nVertex = len(aVertex_in)
    index_vertex = RTreeindex()
    for i in range(nVertex):
        lID = i
        x = aVertex_in[i].dLongitude_degree
        y = aVertex_in[i].dLatitude_degree
        left = x - 1e-5
        right = x + 1e-5
        bottom = y - 1e-5
        top = y + 1e-5
        pBound = (left, bottom, right, top)
        index_vertex.insert(lID, pBound)  #
        pass
    # now the new vertex
    x = pVertex_in.dLongitude_degree
    y = pVertex_in.dLatitude_degree
    delta = max(1e-5, dThreshold_in)
    bbox = (x - delta, y - delta, x + delta, y + delta)
    aIntersect = list(index_vertex.intersection(bbox))
    for k in aIntersect:
        pVertex = aVertex_in[k]
        if pVertex == pVertex_in:
            iFlag_exist = 1
            lIndex = k
            break
        elif (abs(pVertex.dLongitude_degree - x) <= dThreshold_in and
              abs(pVertex.dLatitude_degree - y) <= dThreshold_in):
            iFlag_exist = 1
            lIndex = k
            break

    return iFlag_exist, lIndex

