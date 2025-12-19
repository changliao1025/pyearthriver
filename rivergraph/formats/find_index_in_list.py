import copy
import numpy as np
import importlib.util
from rtree.index import Index as RTreeindex
iFlag_cython = importlib.util.find_spec("cython")

def find_vertex_on_edge(aVertex_in, pEdge_in):
    iFlag_exist = 0
    aIndex= list()
    aIndex_order=list()
    aDistance=list()
    nVertex= len(aVertex_in)
    npoint = 0
    if nVertex > 0 :
        index_vertex = RTreeindex()
        for i in range(nVertex):
            lID = i
            x = aVertex_in[i].dLongitude_degree
            y = aVertex_in[i].dLatitude_degree
            left =   x - 1E-5
            right =  x + 1E-5
            bottom = y - 1E-5
            top =    y + 1E-5
            pBound= (left, bottom, right, top)
            index_vertex.insert(lID, pBound)  #
            pass
        #now the new vertex
        pVertex_start = pEdge_in.pVertex_start
        pVertex_end = pEdge_in.pVertex_end
        x1=pVertex_start.dLongitude_degree
        y1=pVertex_start.dLatitude_degree
        x2=pVertex_end.dLongitude_degree
        y2=pVertex_end.dLatitude_degree
        left   = np.min([x1, x2])
        right  = np.max([x1, x2])
        bottom = np.min([y1, y2])
        top    = np.max([y1, y2])
        pBound= (left, bottom, right, top)
        aIntersect = list(index_vertex.intersection(pBound))
        for k in aIntersect:
            pVertex = aVertex_in[k]
            iFlag_overlap, dDistance, diff = pEdge_in.check_vertex_on_edge(pVertex)
            if iFlag_overlap == 1:
                iFlag_exist = 1
                aDistance.append(dDistance)
                aIndex.append(k)
                npoint = npoint + 1
            else:
                if diff < 1.0:
                    iFlag_overlap = pEdge_in.check_vertex_on_edge(pVertex)

        #re-order, regardless of using rtree or not
        if iFlag_exist == 1 :
            x = np.array(aDistance)
            b = np.argsort(x)
            c = np.array(aIndex)
            d= c[b]
            aIndex_order = list(d)
    else:
        pass

    return iFlag_exist, npoint, aIndex_order
