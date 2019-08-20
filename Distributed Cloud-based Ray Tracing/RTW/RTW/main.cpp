#include <iostream>
#include <fstream>
#include "ray.h"

using namespace std;
/*
unit_vector causes irrational value to occur.
t goes from 1.0 to 0.0
*/
vec3 color(const ray& r) {
	vec3 unit_direction = r.direction();
	double t = 0.5*(unit_direction.y() + 1.0);
	//std::cout << (1.0 - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0) << "\n";
	return (1.0-t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

int main() {
	ofstream img("second.ppm");
	int nx = 200;
	int ny = 100;
	img << "P3\n" << nx << " " << ny << "\n255\n";
	vec3 lower_left_corner(-2.0, -1.0, -1.0);
	vec3 horizontal(4.0, 0.0, 0.0);
	vec3 vertical(0.0, 2.0, 0.0);
	vec3 origin(0.0, 0.0, 0.0);
	for (int j = ny; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			double u = double(i) / double(nx);
			double v = double(j) / double(ny);
			ray r(origin, lower_left_corner + u * horizontal + v * vertical);
			vec3 col = color(r);
			int ir = int(255.99 * col[0]);
			int ig = int(255.99 * col[1]);
			int ib = int(255.99 * col[2]);
			img << ir << " " << ig << " " << ib << "\n";
		}
	}
}