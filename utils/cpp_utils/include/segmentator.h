//
// Created by David Rozenberszki on 05.12.22.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <iostream>
#include <vector>
#include "vec3.h"

// Define structs and operators

// disjoint-set forests using union-by-rank and path compression (sort of).
typedef struct {
    int rank;
    int p;
    int size;
} uni_elt;

typedef struct {
    float w;
    int a, b;
} edge;

bool operator<(const edge& a, const edge& b) {
    return a.w < b.w;
}

typedef struct {
    std::vector<vec3i> m_FaceIndicesVertices;
    std::vector<vec3f> m_Vertices;
    std::vector<vec3f> m_Normals;
    std::vector<vec3f> m_Colors;
} MeshDataf;

// Define the main segmentation function
std::vector<int> segment_mesh(const MeshDataf mesh, const float kthr, const int segMinVerts, std::map<std::pair<int, int>, int>& connectivity);


// Helper functions and classes
class universe {
public:
    universe(int elements) {
        // list of elements
        elts = new uni_elt[elements];
        // number of elements
        num = elements;
        // set initial properties
        for (int i = 0; i < elements; i++) {
            elts[i].rank = 0;
            elts[i].size = 1;
            // position/index of each element
            elts[i].p = i;
        }
    }
    ~universe() { delete[] elts; }
    int find(int x) {
        int y = x;
        // find the
        while (y != elts[y].p)
            y = elts[y].p;
        elts[x].p = y;
        return y;
    }
    int join(int x, int y) {
        // take rank of the element with higher rank
        // combine the sizes
        int final = -1;
        if (elts[x].rank > elts[y].rank) {
            elts[y].p = x;
            elts[x].size += elts[y].size;
            final = x;
        }
        else {
            elts[x].p = y;
            elts[y].size += elts[x].size;
            if (elts[x].rank == elts[y].rank)
                elts[y].rank++;
            final = y;
        }
        // reduce #elements by 1
        num--;
        return final;
    }
    int size(int x) const { return elts[x].size; }
    int num_sets() const { return num; }
private:
    uni_elt* elts;
    int num;
};


vec3f cross(const vec3f& u, const vec3f& v) {
    vec3f c = { u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x };
    float n = sqrtf(c.x * c.x + c.y * c.y + c.z * c.z);
    c.x /= n;  c.y /= n;  c.z /= n;
    return c;
}
vec3f lerp(const vec3f& a, const vec3f& b, const float v) {
    const float u = 1.0f - v;
    return vec3f(v * b.x + u * a.x, v * b.y + u * a.y, v * b.z + u * a.z);
}

inline bool ends_with(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) { return false; }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


