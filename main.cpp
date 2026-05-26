#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <random>
#include <cstring>

#include <vector>
#ifndef M_PI
#define M_PI 3.14159
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"

double sqr(double x) { return x * x; };

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}
double cross(const Vector& a, const Vector& b) {
	return a[0] * b[1] - a[1] * b[0];
}


class Polygon {
public:

    double area() const {
        if (vertices.size() < 3) return 0.0;
        // TODO Lab 2
        // Compute the area of the polygon
        double area = 0.0;
        for(int i = 0; i < vertices.size(); i++){
            const Vector& v1 = vertices[i];
            const Vector& v2 = vertices[(i+1) % vertices.size()];

            area += cross(v1, v2);

        }
        area = fabs(area) * 0.5;
        return area;
    }

    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);
        // TODO Lab 3
        // Compute the centroid of the polygon
        double area = 0.0;
        Vector C(0.0);
        
        for(int i=1; i < vertices.size() - 1; i++){
            const Vector& v1 = vertices[i];
            const Vector& v2 = vertices[(i+1)];
            Vector triangle_centroid = (vertices[0] + v1 + v2) / 3.0;

            double c = cross(v1 - vertices[0], v2 - vertices[0]);
            double T = 0.5 * std::abs(c);
            area += T;

            C = C + T * triangle_centroid;

        }
        if(area < 1e-10) return Vector(0, 0);
        return C / area;

    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;

        // TODO Lab 2
        // Compute the integral of ||x-Pi||^2 over the polygon
        double sum = 0.0;
        for(int i=1; i<vertices.size() - 1; i++){
            Vector T[3] = {vertices[0], vertices[i], vertices[i+1]};
            double abs_T = 0.5 * std::fabs(cross((T[1] - T[0]), (T[2] - T[0])));
            double s = 0.0;
            for(int k=0; k<3; k++){
                for(int l = k; l < 3; l++){
                    s += dot(T[k] - Pi, T[l] - Pi);
                }
            }
            sum += s * abs_T / 6.0;
    }

        return sum;
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}

void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");

    }

    fprintf(f, "</svg>\n");
    fclose(f);
}

bool inside(Vector X, const Vector P0, const Vector Pi, double w0, double wi){
        return (X - P0).norm2() - w0 <= (X - Pi).norm2() - wi;
    }


class VoronoiDiagram {

public:

    VoronoiDiagram() {
    };


    void compute() {

        // TODO Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi
    
        cells.clear();
        cells.resize(points.size());

        if(!use_partial){
            weights.resize(points.size());;
        }
        else{
            if(weights.size() != points.size() + 1){
                weights.resize(points.size() + 1, 0.0);
            }
        }


        for(int i=0; i<points.size(); i++){
            Polygon cell;
            cell.vertices.push_back(Vector(0, 0));
            cell.vertices.push_back(Vector(1, 0));
            cell.vertices.push_back(Vector(1, 1));
            cell.vertices.push_back(Vector(0, 1));

            for(int j=0; j< points.size(); j++){
                if(i==j){
                    continue;
                }
                cell = clip_by_bisector(cell, points[i], points[j], weights[i], weights[j]);
                if(cell.vertices.empty()) break;
            }
            if(use_partial && !cell.vertices.empty()){
                double w_air = weights[points.size()];
                if(weights[i] > w_air){
                    double r2 = std::sqrt(weights[i] - w_air);

                    for(int k=0;k<disk_samples;k++){
                    double theta0 = 2.0 * M_PI * k / disk_samples;
                    double theta1 = 2.0 * M_PI * (k+1) / disk_samples;

                    Vector u(points[i][0] + r2*std::cos(theta0), points[i][1] + r2*std::sin(theta0));
                    Vector v(points[i][0] + r2*std::cos(theta1), points[i][1] + r2*std::sin(theta1));

                    cell = clip_by_edge(cell, u, v);

                    if(cell.vertices.empty()) break;
                }
                }
                 else{
                cell.vertices.clear();
            }
        }
            cells[i] = cell;
            
    }
}


    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk

        Polygon result;
        Vector edge = v - u;
        double len = std::sqrt(edge[0] * edge[0] + edge[1] * edge[1]);
        Vector n(-edge[1] / len, edge[0] / len);

        auto inside_edge = [&](const Vector& P){
            return dot(P - u, n) >= 0;
        };

        for(int i=0; i<V.vertices.size(); i++){
            Vector A = V.vertices[(i > 0) ? i - 1 : V.vertices.size() - 1];
            Vector B = V.vertices[i];

            bool A_inside = inside_edge(A);
            bool B_inside = inside_edge(B);

            if(B_inside){
                if(!A_inside){
                    double denom = dot(B - A, n);
                    Vector P = A + (dot(u - A, n) / denom) * (B-A);
                    result.vertices.push_back(P);
                }
                result.vertices.push_back(B);
            }
            else if(A_inside){
                double denom = dot(B - A, n);
                Vector P = A + (dot(u - A, n) / denom) * (B-A);
                result.vertices.push_back(P);
            }
        }

        return result;
    }
    Vector intersect_edge(const Vector& A, const Vector& B, const Vector& u, const Vector& v){
        Vector edge  = v - u;
        double len = std::sqrt(edge[0] * edge[0] + edge[1] * edge[1]);
        Vector n(-edge[1] / len, edge[0] / len);

        double denom = dot(B - A, n);
        if(std::abs(denom) < 1e-20) return A;

        return A + (dot(u - A, n) / denom) * (B-A);

    }
    static Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi, double w0, double wi) {

        // TODO Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        // TODO Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1

        Polygon result;

        int n = V.vertices.size();

        for(int i=0; i < n ; i++){
            Vector curVertex = V.vertices[i];
            Vector preVertex = V.vertices[(i>0)?(i-1):(n-1)];
            Vector M = (Pi + P0) / 2;

            Vector M_prime = M + ((w0 - wi) / (2 * (P0 - Pi).norm2()) * (Pi - P0));

            Vector A = preVertex;
            Vector B = curVertex;

            bool A_inside = inside(A, P0, Pi, w0, wi);
            bool B_inside = inside(B, P0, Pi, w0, wi);

            if(B_inside){
                if(!A_inside){

                    double denom = dot(B - A, Pi- P0);
                    if(std::abs(denom) > 1e-20){
                        double t = dot(M_prime-A, Pi-P0) / denom;
                        Vector P = A + t*(B-A);
                        result.vertices.push_back(P);
                    }
                }
                result.vertices.push_back(B);
            }
            else if (A_inside){
                    double denom = dot(B - A, Pi- P0);
                    if(std::abs(denom) > 1e-20){
                        double t = dot(M_prime-A, Pi-P0) / denom;
                        Vector P = A + t*(B-A);
                        result.vertices.push_back(P);
                    }
                }
        
        }

        return result;
    }


    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    std::vector<Polygon> cells;   // Lab 1 : the polygons representing each individual cell

    bool use_partial = false;
    int disk_samples = 32;
};


// Lab 2 
class OptimalTransport {

public:
    OptimalTransport() {};

    void optimize();

    VoronoiDiagram vor;
    double fluid_volume = 0.4;
    bool use_partial = false;
    int disk_samples = 32;
};


// Labs 2 and 3
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);

    // first compute the Voronoi diagram at the current optimization step
    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
     int N = ot->vor.points.size();
    
    ot->vor.compute();
  
   
    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)
    
    lbfgsfloatval_t fx = 0.0;
    double ddesired_fluid_volume = ot->fluid_volume;
    double desired_air_volume = 1.0 - ddesired_fluid_volume;

    double estimated_fluid_volume = 0.0;

    double target = ddesired_fluid_volume / (double)ot->vor.points.size();
    // g[i] = 
    // fx = ...
    for(int i=0; i < N; i ++){
        double area_i = ot->vor.cells[i].area();
        double weight_i = ot->vor.weights[i];
        
        double integral_i = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]) - weight_i * area_i;
        double extra_i = target * weight_i;
        estimated_fluid_volume += area_i;
        fx += integral_i + extra_i;
    }
    double estimated_air_volume = 1.0 - estimated_fluid_volume;
    double w_air = ot->vor.weights[ot->vor.weights.size() - 1];
    fx += w_air * (desired_air_volume - estimated_air_volume);

    for(int i=0; i<ot -> vor.cells.size(); i++){
        g[i] = ot->vor.cells[i].area() - target;
    }

    g[ot->vor.weights.size() - 1] = estimated_air_volume - desired_air_volume;

    return -fx;

}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(
    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls) {
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}


// Lab 2
void OptimalTransport::optimize() {

    lbfgsfloatval_t fx;
    std::vector<double> weights(vor.weights);

    lbfgs_parameter_t param;
    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    // run the LBFGS optimizer
    int ret = lbfgs(weights.size(), &weights[0], &fx, evaluate, progress, (void*)this, &param);

    // copy the result back to the voronoi structure
    vor.weights = weights;

    // finally recompute the Voronoi diagram with the final optimized weights
    vor.compute();
}


// Lab 3 (fluids)
class Fluid {
public:
    Fluid(int N_particles = 1000) : N_particles(N_particles) {

        fluid_volume = 0.25;
        particles.clear();
        velocities.clear();

        particles.reserve(N_particles);
        velocities.reserve(N_particles);

        std::default_random_engine gen(42);
        std::uniform_real_distribution<double> U(0.0, 1.0);

        for(int i=0; i<N_particles; i++){
            double x = 0.15 + 0.3 * U(gen);
            double y = 0.10 + 0.7 * U(gen);

            particles.push_back(Vector(x, y));
            velocities.push_back(Vector(0, 0));
        }

        ot.vor.points = particles;
        ot.vor.weights.assign(N_particles +  1, 1.0);
        double r2 = fluid_volume / (M_PI * N_particles);
        ot.vor.weights[N_particles] = 1.0 - r2;

        ot.vor.disk_samples = 48;
        ot.fluid_volume = fluid_volume;
        ot.vor.compute();
        ot.optimize();
    }

    // Lab 3 : advance the simulation dt in time
    void time_step(double dt) {

        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81);
        double m_i = 200;

        // TODO Lab 3 : 
        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell centroid, integrate acceleration->velocity and velocity->position
 
        ot.vor.points = particles;
        ot.fluid_volume = fluid_volume;
        ot.vor.use_partial = true;

        ot.optimize();

        for(int i=0; i<N_particles;i++){
            if(ot.vor.cells[i].vertices.size() < 3) continue;

            Vector C = ot.vor.cells[i].centroid();

            Vector F_spring = (C - particles[i]) / epsilon2;
            Vector F = F_spring + m_i * g;

            Vector acceleration = F / m_i;

            velocities[i] = velocities[i] + dt * acceleration;

            velocities[i] = 0.999 * velocities[i];

            particles[i] = particles[i] + dt * velocities[i];

            double bounce = 0.5;

            if(particles[i][0] < 0.0){
                particles[i][0] = 0.0;
                velocities[i][0] = -bounce * velocities[i][0];
            }
            if(particles[i][0] > 1.0){
                particles[i][0] = 1.0;
                velocities[i][0] = -bounce * velocities[i][0];
            }
            if(particles[i][1] < 0.0){
                particles[i][1] = 0.0;
                velocities[i][1] = -bounce * velocities[i][1];
            }

            if(particles[i][1] > 1.0){
                particles[i][1] = 1.0;
                velocities[i][1] = -bounce * velocities[i][1];
            }
        }

        ot.vor.points = particles;
        ot.vor.compute();
    
    }

    // just run the full simulation
    void run_simulation() {
        double dt = 0.002;
        for (int i = 0; i < 1000; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "test", i);
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double fluid_volume; // you decide the fraction of the unit square occupied by the fluid
};



int main() {

    int N = 100;
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double>U(0.0, 1.0);


    /*Polygon p;
    p.vertices.push_back(Vector(0.1, 0.2));
    p.vertices.push_back(Vector(0.6, 0.4));
    p.vertices.push_back(Vector(0.5, 0.7));
    p.vertices.push_back(Vector(0.2, 0.5));

    std::vector<Polygon> s;
    s.push_back(p);

    save_frame(s, "toto");
    save_svg(s, "toto.svg");*/

    /*VoronoiDiagram vor;
    vor.points.push_back(Vector(0.1, 0.2));
    vor.points.push_back(Vector(0.6, 0.4));
    vor.points.push_back(Vector(0.5, 0.7));
    vor.points.push_back(Vector(0.2, 0.5));

    vor.compute();

    OptimalTransport ot;

    for(int i=0; i<N; i++){
        double x = U(gen);
        double y = U(gen);

        ot.vor.points.push_back(Vector(x, y));
    }

    ot.vor.weights.assign(N, 0.0);

    ot.vor.weights.resize(ot.vor.points.size(), 0.0);

    ot.optimize();
    

    save_frame(ot.vor.cells, "lab2");
    save_svg(ot.vor.cells, "lab2.svg", &ot.vor.points, "white");*/

    Fluid fluid(300);
    fluid.run_simulation();
    return 0;
}

/*
clang -I. -c lbfgs.c -o lbfgs.o
clang++ -std=c++11 -I. main.cpp lbfgs.o -o project2
*/