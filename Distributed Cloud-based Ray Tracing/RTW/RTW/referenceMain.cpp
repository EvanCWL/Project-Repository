#include <fstream>

using namespace std;
const int width = 255, height = 255;

int main() {
	ofstream img("picture.ppm");
	img << "P3" << endl;
	img << width << " " << height << endl;
	img << "255" << endl;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int r = x % 255;
			int g = y % 255;
			int b = y*x % 255;
			img << r << " " << g << " " << b << endl;
		}
	}
}