#include <iostream>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <unistd.h>

using namespace std;
  
class GameOfLife
{
	public:
		GameOfLife(int height, int width);
		void show();
		void update();
		void iterate(unsigned int iterations);
		char getState(int x, int y); 
	private:
		char **world1;
		char **world2;
		int height;
		int width;
		bool flag;
};
 
GameOfLife::GameOfLife(int height, int width) 
{
	this->height = height;
	this->width = width;
	flag = true;
	world1 = new char * [height];
	world2 = new char * [height];
	world1[0] = new char[height * width];
	world2[0] = new char[height * width];

	for (int i = 1; i < height; i++)
	{
		world1[i] = world1[i - 1] + width;
		world2[i] = world2[i - 1] + width;
	}
	for (int i = 0; i < height; i++)
        	for (int j = 0; j < width; j++)
		{
			if ((rand() % 3 + 1) == 3)
            			world1[i][j] = 'o';
			else
				world1[i][j] = '-';
			world2[i][j] = '-';
		}
}

void GameOfLife::show() 
{
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++)
			if (flag)
				cout << world1[i][j];
			else
				cout << world2[i][j];
		cout << endl;
	}
}
 
void GameOfLife::update() {
	for (int i = 0; i < height; i++ ) 
		for (int j = 0; j < width; j++ ) 
			if (flag)
				world2[i][j] = GameOfLife::getState(i, j);
			else
				world1[i][j] = GameOfLife::getState(i, j);
	flag = !flag;
}
 
char GameOfLife::getState(int x, int y) 
{
	int d[3] = {-1, 0, 1};
	int neighbors = 0;
	
	for (int i = 0; i < 3; i++) 
		for (int j = 0; j < 3; j++) 
			if (d[i] != 0 || d[j] != 0) 
				if ((x + d[i] >= 0 && x + d[i] < height) && (y + d[j] >= 0 && y + d[j] < width))
					if (flag)
					{
						if (world1[x + d[i]][y + d[j]] == 'o')
							neighbors++;
					}
					else
					{
						if (world2[x + d[i]][y + d[j]] == 'o')
							neighbors++;
					}
	if (flag)
	{
		if (world1[x][y] == 'o') 
			return (neighbors > 1 && neighbors < 4) ? 'o' : '-';
		else
			return (neighbors == 3) ? 'o' : '-';
	}
	else
	{
		if (world2[x][y] == 'o') 
			return (neighbors > 1 && neighbors < 4) ? 'o' : '-';
		else
			return (neighbors == 3) ? 'o' : '-';
	}
}
 
void GameOfLife::iterate( unsigned int iterations ) 
{
	for ( int i = 0; i < iterations; i++ )
	{
		show();
		printf("\033c");
		usleep(100000);
		update();
	}
	show();
}
 
int main(int argc, const char **argv) 
{	
	srand (time(NULL));

	if (argc == 4)
	{
		int height;
		int width;
		int iterations;

		height = atoi(argv[1]);
		width = atoi(argv[2]);
		iterations = atoi(argv[3]);
		GameOfLife gol(height, width);

		gol.iterate(iterations);	
	}
	else
	{
		cout << "Incorrect input!\nPlease try again..." << endl;
		return -1;
	}	
	return 0;
}
