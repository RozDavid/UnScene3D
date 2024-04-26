//
// Created by David Rozenberszki on 05.12.22.
//


#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <pybind11/stl.h>
#include "include/segmentator.h"

using std::vector;
using std::string;

// Felzenszwalb segmentation (https://cs.brown.edu/~pff/segment/index.html)


universe* segment_graph(int num_vertices, int num_edges, edge* edges, float c) {
    std::sort(edges, edges + num_edges);  // sort edges by weight
    universe* u = new universe(num_vertices);  // make a disjoint-set forest
    // threshold for each vertex
    float* threshold = new float[num_vertices];
    // set initial threshold
    for (int i = 0; i < num_vertices; i++) {
        threshold[i] = c;
    }
    // for each edge, in non-decreasing weight order
    for (int i = 0; i < num_edges; i++) {
        edge* pedge = &edges[i];
        // components conected by this edge = initially 2 vertices
        int a = u->find(pedge->a);
        int b = u->find(pedge->b);
        if (a != b) {
            if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) {
                // join components
                u->join(a, b);
                a = u->find(a);
                // update threshold for a = edge weight + kthr / |a|
                threshold[a] = pedge->w + (c / u->size(a));
            }
        }
    }
    delete[] threshold;
    return u;
}


vector<int> segment_mesh(MeshDataf mesh, const float kthr, const int segMinVerts, std::map<std::pair<int, int>, int>& connectivity) {

    // Pass mesh file to C++ function
    // vertices should be of c++ float type with size (num_verts, 3)
    // faces should be of c++ int type with size (num_faces, 3)
    // Normals can be initialized here as cleared anyway originally

    size_t edgesCount = mesh.m_FaceIndicesVertices.size() * 3;
    edge* edges = new edge[edgesCount];
    vector<int> counts(mesh.m_Vertices.size(), 0);
    // initialize with zeros
    mesh.m_Normals.clear();
    mesh.m_Normals = std::vector<vec3f>(mesh.m_Vertices.size());

    // Compute face normals and smooth into vertex normals
    for (int i = 0; i < mesh.m_FaceIndicesVertices.size(); i++) {
        const uint32_t i1 = mesh.m_FaceIndicesVertices[i][0];
        const uint32_t i2 = mesh.m_FaceIndicesVertices[i][1];
        const uint32_t i3 = mesh.m_FaceIndicesVertices[i][2];

        vec3f p1 = mesh.m_Vertices[i1];
        vec3f p2 = mesh.m_Vertices[i2];
        vec3f p3 = mesh.m_Vertices[i3];

        const int ebase = 3 * i;

        edges[ebase].a = i1;  edges[ebase].b = i2;
        edges[ebase + 1].a = i1;  edges[ebase + 1].b = i3;
        edges[ebase + 2].a = i3;  edges[ebase + 2].b = i2;

        // smoothly blend face normals into vertex normals
        vec3f normal = cross(p2 - p1, p3 - p1);
        mesh.m_Normals[i1] = lerp(mesh.m_Normals[i1], normal, 1.0f / (counts[i1] + 1.0f));
        mesh.m_Normals[i2] = lerp(mesh.m_Normals[i2], normal, 1.0f / (counts[i2] + 1.0f));
        mesh.m_Normals[i3] = lerp(mesh.m_Normals[i3], normal, 1.0f / (counts[i3] + 1.0f));
        counts[i1]++; counts[i2]++; counts[i3]++;
    }

    // std::cout << "Constructing edge graph based on mesh connectivity..." << std::endl;
    for (int i = 0; i < edgesCount; i++) {
        int a = edges[i].a;
        int b = edges[i].b;

        vec3f& n1 = mesh.m_Normals[a];
        vec3f& n2 = mesh.m_Normals[b];
        vec3f& p1 = mesh.m_Vertices[a];
        vec3f& p2 = mesh.m_Vertices[b];

        // get the edge as a vector
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float dz = p2.z - p1.z;
        // normalize its length
        float dd = sqrtf(dx * dx + dy * dy + dz * dz);
        dx /= dd; dy /= dd; dz /= dd;
        // similarity of vertex normals
        float dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
        // distance between normals
        float normal_dist = 1.0f - dot;

        vec3f& color1 = mesh.m_Colors[a];
        vec3f& color2 = mesh.m_Colors[b];
        vec3f colordiff = math::abs(color1 - color2);
        float color_dist = (colordiff[0] + colordiff[1] + colordiff[2]);
        float dist = normal_dist * color_dist;

        // angle between vertex normal and edge
        float dot2 = n2.x * dx + n2.y * dy + n2.z * dz;
        // too much? make weight smaller, only if colors are similar
        if (dot2 > 0 && color_dist < 0.05) {
            // make it much less of a problem if convex regions have normal difference
            dist = dist * dist;
        }
        // set edge weight ~ dissimilarity
        edges[i].w = dist;
    }

    // Segment!
    universe* u = segment_graph(mesh.m_Vertices.size(), edgesCount, edges, kthr);

    // Joining small segments
    for (int j = 0; j < edgesCount; j++) {
        int a = u->find(edges[j].a);
        int b = u->find(edges[j].b);
        if ((a != b) && ((u->size(a) < segMinVerts) || (u->size(b) < segMinVerts))) {
            u->join(a, b);
        }
    }

    // Return segment indices as vector
    vector<int> outComps(mesh.m_Vertices.size());
    for (int q = 0; q < mesh.m_Vertices.size(); q++) {
        outComps[q] = u->find(q);
    }

    for (int i = 0; i < edgesCount; i++) {

        int s1 = u->find(edges[i].a);
        int s2 = u->find(edges[i].b);
        std::pair<int, int> key(s1, s2);
        if (connectivity.count(key) == 0 && s1 != s2)
            connectivity.insert ( std::pair<std::pair<int, int>, int>(key, 0) );
    }

    delete edges;
    delete u;
    return outComps;
}


// Wrapper function to first process input data
namespace py = pybind11;
py::tuple segment_mesh_wrapper(py::array np_vertices, py::array np_faces, py::array np_colors, const float kthr=0.005, const int segMinVerts=20){

    // get sizes of objects
    auto castedVertices = np_vertices.request();
    auto castedFaces = np_faces.request();
    auto castedColors = np_colors.request();
    uint32_t nVertices = castedVertices.shape[0];
    uint32_t nFaces = castedFaces.shape[0];

    // Request buffers
    float* request_vertices = (float*) castedVertices.ptr;
    int* request_faces = (int*) castedFaces.ptr;
    float* request_colors = (float*) castedColors.ptr;

    // allocate and parse vectors
    vector<vec3f> vertices(nVertices, vec3f(0., 0., 0.));
    vector<vec3f> normals(nVertices, vec3f(0., 0., 0.));
    vector<vec3f> colors(nVertices, vec3f(0., 0., 0.));
    vector<vec3i> faces(nFaces, vec3f(0., 0., 0.));

    for (int i = 0; i < nVertices; ++i) {
        vertices[i] = vec3f(request_vertices[i * 3], request_vertices[i * 3 + 1], request_vertices[i * 3 + 2]);
        colors[i] = vec3f(request_colors[i * 3], request_colors[i * 3 + 1], request_colors[i * 3 + 2]);
    }
    for (int i = 0; i < nFaces; ++i) {
        faces[i] = vec3i(request_faces[i * 3], request_faces[i * 3 + 1], request_faces[i * 3 + 2]);
    }

    // Parse to MeshDataf object
    // std::cout << "Segmenting: " << meshFile << ", threshold: " << kthr << ", min verts: " << segMinVerts << std::endl;
    MeshDataf mesh;
    mesh.m_Vertices = vertices;
    mesh.m_Normals = normals;
    mesh.m_Colors = colors;
    mesh.m_FaceIndicesVertices = faces;

    // Do the actual processing
    // std::cout << "Starting segmentation with " << nVertices << " vertices and " << nFaces << " faces." << std::endl;
    std::map<std::pair<int, int>, int> connectivity;
    const vector<int> comps = segment_mesh(mesh, kthr, segMinVerts, connectivity);

    // Map comps and connectivity to a [0-N] range instead of vertex indices
    std::map<int, int> comp_mapping;
    vector<int> comps_new;
    comps_new = comps;  // copy by value

    // Get sorted unique components
    sort(comps_new.begin(), comps_new.end());
    auto last = unique(comps_new.begin(), comps_new.end());
    comps_new.erase(last, comps_new.end());

    // Add components to mapping
    for (int unique_index = 0; unique_index < comps_new.size(); unique_index++) {
        comp_mapping.insert(std::pair<int, int>(comps_new[unique_index], unique_index));
    }

    // Map components
    comps_new = comps;  // copy by value
    for (int i = 0; i < comps.size(); i++) {
        comps_new[i] = comp_mapping[comps[i]];
    }

    // Map connectivity
    std::map<std::pair<int, int>, int> connectivity_new;
    for(std::map<std::pair<int, int>, int>::iterator iter = connectivity.begin(); iter != connectivity.end(); ++iter)
    {
        std::pair<int, int> key = iter->first;
        std::pair<int, int> key_new(comp_mapping[key.first], comp_mapping[key.second]);
        connectivity_new.insert(std::pair<std::pair<int, int>, int>(key_new, 0));
    }

    // Cast connectivity vector into 2D array  - there must be an easier way to do it, but this works
    int num_segment_pairs = connectivity_new.size();
    py::array_t<int> np_connectivity = py::array_t<int>(num_segment_pairs * 2);
    auto np_connectivity_buff = np_connectivity.request();
    int* buff_ptr = (int*) np_connectivity_buff.ptr;
    int index = 0;
    for(std::map<std::pair<int, int>, int>::iterator iter = connectivity_new.begin(); iter != connectivity_new.end(); ++iter)
    {
        buff_ptr[index * 2] = iter->first.first;
        buff_ptr[index * 2 + 1] = iter->first.second;
        index++;
    }
    np_connectivity.resize({num_segment_pairs, 2});

    // Cast components into single array
    py::array np_comps =  py::cast(comps_new);

    return py::make_tuple(np_comps, np_connectivity);



}

PYBIND11_MODULE(felzenszwalb_cpp, m) {
    m.doc() = "Wrapping the Felzenszwalb segmentation function with numpy ";
    m.def("segment_mesh", &segment_mesh_wrapper, "Oversegmentation on a mesh to geometrically consistent segments, returning their ids as numpy array");
}
