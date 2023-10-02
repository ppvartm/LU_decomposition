#pragma once
#include <ctime>
#include <iostream>
#include <string>


struct timer
{
	clock_t time1 = 0;
	clock_t time2 = 0;
	std::string log;
	timer(const std::string& log_) :log(log_)
	{
		time1 = clock();
	}
	~timer()
	{
		time2 = clock();
		std::cout << "Execution time of " << log << " is " << static_cast<double>(time2 - time1) / CLOCKS_PER_SEC << "\n";
	}
};

