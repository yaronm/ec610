#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <queue>
#include <math.h>
#include <assert.h>
#include<climits>
#include <chrono>

enum event_type { a, o, d };
struct event_id { event_type t; int d_num; int skip; };
struct event_occurence { event_id e; double t; };

/////////////////////////////////////////////////////////
/////////////////////////Q1//////////////////////////////
/////////////////////////////////////////////////////////

void exp_rv(std::vector<double> &vals, double &mu_inv) {
	/*
	* generate a distribution according to an exponential random variable, with parameter mu_inv
	* vals contains the distribution
	*/
	//mu_inv = 1/mu
	std::random_device rand;
	std::mt19937 gen(rand());
	std::uniform_real_distribution<>dis(0, 1.0);//to be used to get uniform distribution


	for (int n = 0; n<1000; ++n) {
		double val1 = dis(gen);//get next value in uniform distribution
		double val = log(1 - val1)*-1 * mu_inv;//mu_inv might need to be divided, need to talk to prof to find out exactly what mu_inv is
		vals.push_back(val);//store exponential distribution value in vals
	}
}

int get_expoential_q1() {
	/*
	answer to q1, gets 1000 values according to an exponential distribution
	prints out the mean and variance
	*/
	std::vector<double> vals;
	double variance = 0;
	double mean = 0;
	double mu_inv = 10;
	exp_rv(vals, mu_inv);
	//get the mean
	for (double val : vals) {
		mean += val;
	}
	mean /= 1000;

	//get the variance
	for (double val : vals) {
		variance += pow((val - mean), 2);
	}
	variance /= 1000;
	std::cout << mean << "," << variance << std::endl;
	return 0;
}



/////////////////////////////////////////////////////////
/////////////////////////Q1 done/////////////////////////
/////////////////////////////////////////////////////////

event_occurence create_event(event_type et, int o, int skip, double &t) {
	/*creates an event occurence*/
	event_type e = et;
	event_id ei;
	ei.t = e;
	ei.d_num = o;
	ei.skip = skip;
	event_occurence eo;
	eo.e = ei;
	eo.t = t;//store departure time
	return eo;
}

//create distributions

int generate_poisson(double & T, std::queue<double> &times, double &mu, unsigned seed) {
	/*
	generate a poisson distribution with parameter mu. generates values until
	the event time is greater than T and stores the values in times.
	used for arrival = M and observations
	*/
	double start = 0;
	int counter = 0;

	std::mt19937 gen(seed);
	std::uniform_real_distribution<>dis(0, 1.0);
	while (start <= T) {
		start += log(1 - dis(gen))*-1 / mu; //interarrival time of poisson distribution is exponentially distributed
		if (start <= T) {
			times.push(start);
			counter++;
		}
	}
	return counter;
}

int generate_deterministic_arrival(double & T, std::queue<double> &times, double &mu, unsigned seed) {
	/*
	generate a deterministic distribution with parameter mu. generates values until
	the event time is greater than T and stores the values in times.
	used for arrival = D
	*/
	double start = 0;
	int counter = 0;
	while (start <= T) {
		start += mu; //interarrival time is deterministically distributed
		if (start <= T) {
			times.push(start);
			counter++;
		}
	}
	return counter;
}

void generate_exponential(int num, std::queue<double> &lengths, double &C, double& unuses, double &mu, double &unused, double &unused2, unsigned seed) {
	/*
	generates num values according to an exponential distribution with parameter mu
	stores these values in lengths.
	To be used for exponential service time
	*/
	std::mt19937 gen(seed);
	std::uniform_real_distribution<>dis(0, 1.0);
	for (int counter = 0; counter<num; counter++) {
		lengths.push(log(1 - dis(gen))*-1 / mu);
	}
}

void generate_bipolar(int num, std::queue<double> &lengths, double &C, double &p, double& unused, double& L1, double &L2, unsigned seed) {
	/*
	generates num values according to a bipolar distribution with parameter p
	and lengths L1 and L2. Stores these values in lengths.
	To be used for G service
	*/

	std::mt19937 gen(seed);
	std::uniform_real_distribution<>dis(0, 1.0);
	for (int counter = 0; counter<num; counter++) {
		if (dis(gen) <= (1 - p)) {
			lengths.push(C / L2);
		}
		else {
			lengths.push(C / L1);
		}
	}
}

void generate_deterministic_service(int num, std::queue<double> &lengths, double &C, double &unuses, double &mu, double &unused, double &unused2, unsigned seed) {
	/*
	generates num values according to a deterministic distribution with parameter mu
	stores these values in lengths.
	To be used for deterministic service time
	*/

	for (int counter = 0; counter<num; counter++) {
		lengths.push(mu);
	}
}

//end of distribution generation functions

//create event queues

double arrival_departures_1_server_infinite_queue(std::queue<double> &arrivals, std::queue<double> &departures,
	double &C, std::queue<event_occurence> &ad, std::queue<double> &observations) {
	/*
	uses a vector of arrivals and departures to create the departure times when
	there is 1 server and place arrivals and departures in
	order. also places the observations in order this is an infinite buffer
	returns the number of packets lost/number of packets processed
	//should not be used, is here for purposes of 1 of questions
	*/
	int a_index = 0;//how many arrivals we have added to the queue
	int d_index = 0;//which departure we are adding to the queue
	int o_index = 0;//which observation we are adding to the queue

	double next_departure = 0; //next departure time, 0 =>no packet is being serviced
	double last_departure = 0; //most recent departure time
	double next_arrival = 0;
	double next_observation = 0;

	while (!arrivals.empty() || !observations.empty() || next_arrival != 0 || next_observation != 0 || d_index <a_index) {
		if (next_arrival == 0 && !arrivals.empty()) {
			next_arrival = arrivals.front();
			arrivals.pop();
		}
		if (next_observation == 0 && !observations.empty()) {
			next_observation = observations.front();
			observations.pop();
		}
		event_occurence eo;
		if (next_departure != 0 && ((next_arrival != 0 && next_observation != 0
			&& next_departure < next_arrival && next_departure < next_observation)
			|| (next_arrival == 0 && next_departure <next_observation) ||
			(next_observation == 0 && next_departure<next_arrival) ||
			(next_observation == 0 && next_arrival == 0))) {
			d_index++;//start counting events at 1, so increment first
					  //create a new departure event
			eo = create_event(d, d_index, 0, next_departure);
			last_departure = next_departure;//we need this time to determine when we start processing
											//the next packet
			next_departure = 0;
			if (d_index < a_index) {//we have packets in the queue
				next_departure = last_departure + (1 / departures.front());//get next departure time
				departures.pop();
			}
		}
		else if (next_arrival != 0 && (next_arrival < next_observation || next_observation == 0)) {
			a_index++;
			eo = create_event(a, a_index, 0, next_arrival);
			if (next_departure == 0) {//if the system was empty before this arrival, this is the next
									  //parcket to be serviced
				next_departure = next_arrival + (1 / departures.front());
				departures.pop();
			}
			next_arrival = 0;
		}
		else {
			o_index++;
			eo = create_event(o, o_index, 0, next_observation);
			next_observation = 0;
		}
		ad.push(eo);
	}
	return 0;
}

event_occurence add_departure_event_finite_queue(int &d_index, int &skip_next, double &next_departure, double & last_departure, double &C,
	std::queue<int> &dropped, int &a_index, int &next_to_skip, std::queue<double>& departures) {

	d_index++;//start counting events at 1, so increment first

			  //create a new departure event
	event_occurence eo = create_event(d, d_index, skip_next, next_departure);

	if (skip_next == 0) {
		last_departure = next_departure;//we need this time to determine when we start processing
										//the next packet  
	}

	next_departure = 0;

	if (d_index<a_index) {
		if (!dropped.empty() && (d_index + 1) == next_to_skip) {
			next_departure = last_departure + (1 / departures.front());//get next departure time
			departures.pop();
			skip_next = 1;
			next_to_skip = dropped.front();
			dropped.pop();
			if (dropped.empty()) {
				next_to_skip = -1;
			}
		}
		else {///we have packets in the queue
			next_departure = last_departure + (1 / departures.front());//get next departure time
			departures.pop();
			skip_next = 0;
		}
	}
	return eo;
}

double arrival_departures_1_server(std::queue<double> &arrivals, std::queue<double>& departures,
	double& C, int K, std::queue<event_occurence> &ad, std::queue<double> &observations) {
	/*
	uses a vector of arrivals and departures to create the departure times when
	there is a queue of size K and 1 server and place arrivals, observations  and departures in
	order. if K= -1, this is an infinite buffer
	returns the number of packets lost/number of packets processed
	*/
	int a_index = 0;//how many arrivals we have added to the queue
	int d_index = 0;//which departure we are adding to the queue
	int o_index = 0;//which observation we are adding to the queue
	double next_departure = 0; //next departure time, 0 =>no packet is being serviced
	double last_departure = 0; //most recent departure time
	double next_arrival = 0;
	double next_observation = 0;
	double num_dropped = 0;
	int skip_next = 0;
	int next_to_skip = -1;
	std::queue<int> dropped;
	while (!arrivals.empty() || !observations.empty() || next_arrival != 0 || next_observation != 0 || d_index < a_index) {
		if (next_arrival == 0 && !arrivals.empty()) {
			next_arrival = arrivals.front();
			arrivals.pop();
		}
		if (next_observation == 0 && !observations.empty()) {
			next_observation = observations.front();
			observations.pop();
		}
		event_occurence eo;
		if (next_departure != 0 &&
			((next_arrival != 0 && next_observation != 0
				&& next_departure < next_arrival && next_departure < next_observation)
				|| (next_arrival == 0 && next_departure <next_observation) ||
				(next_observation == 0 && next_departure<next_arrival) ||
				(next_observation == 0 && next_arrival == 0))) {
			eo = add_departure_event_finite_queue(d_index, skip_next, next_departure, last_departure, C, dropped, a_index, next_to_skip, departures);
		}
		else if (next_arrival != 0 && (next_arrival < next_observation || next_observation == 0)) {
			a_index++;
			eo = create_event(a, a_index, 0, next_arrival);
			if (K >= 0 && a_index - d_index - K - num_dropped >3)//1 for the server, 1 because we have incremented a_index, 1 because
			{//we have K spots, so it is when we have K+1 waiting that we have a problem
				num_dropped++;
				if (dropped.empty() || next_to_skip == -1) {
					next_to_skip = a_index;
				}
				else {
					dropped.push(a_index);
				}
				eo.e.skip = 1;
			}
			else {
				if (next_departure == 0) {//if the system was empty before this arrival, this is the next
										  //parcket to be serviced
					next_departure = next_arrival + (1 / departures.front());
					departures.pop();
				}
			}
			next_arrival = 0;
		}
		else {
			o_index++;
			eo = create_event(o, o_index, 0, next_observation);
			next_observation = 0;
		}
		if (eo.e.t == d && eo.e.skip == 1)//don't add in dropped packets' departure times
			continue;
		ad.push(eo);
	}
	return num_dropped / a_index;
}

double arrival_departures_2_server_infinite_buffer(std::queue<double>& arrivals, std::queue<double> &departures,
	double& C, std::queue<event_occurence> &ad, std::queue<double>& observations) {
	/*
	uses a vector of arrivals and departures to create the departure times when
	there is an infinite queue and 2 servers and place arrivals, observations and departures in
	order
	*/
	int a_index = 0;//how many arrivals we have added to the queue
	int d_index = 0;//which departure we are adding to the queue
	int o_index = 0;//which observation we are adding to the queue

	double next_departure1 = 0; //next departure time, 0 =>no packet is being serviced
	double next_departure2 = 0;
	double last_departure1 = 0; //most recent departure time
	double last_departure2 = 0;
	double next_arrival = 0;
	double next_observation = 0;

	double * next_departure = &next_departure1;
	double *last_departure = &last_departure1;
	while (!arrivals.empty() || !observations.empty() || next_arrival != 0 || next_observation != 0 || d_index < a_index) {
		if (next_arrival == 0 && !arrivals.empty()) {
			next_arrival = arrivals.front();
			arrivals.pop();
		}
		if (next_observation == 0 && !observations.empty()) {
			next_observation = observations.front();
			observations.pop();
		}
		event_occurence eo;
		if (next_departure1 != 0 && ((next_arrival != 0 && next_observation != 0
			&& next_departure1 < next_arrival && next_departure1 < next_observation)
			|| (next_arrival == 0 && next_departure1 < next_observation) ||
			(next_observation == 0 && next_departure1 < next_arrival) ||
			(next_observation == 0 && next_arrival == 0)) ||
			next_departure2 != 0 && ((next_arrival != 0 && next_observation != 0
				&& next_departure2 < next_arrival && next_departure2 < next_observation)
				|| (next_arrival == 0 && next_departure2 < next_observation) ||
				(next_observation == 0 && next_departure2 < next_arrival) ||
				(next_observation == 0 && next_arrival == 0)))
		{
			if ((next_departure1 != 0 && next_departure1 < next_departure2) || next_departure2 ==0) {
				next_departure = &next_departure1;
				last_departure = &last_departure1;
			}
			else {
				next_departure = &next_departure2;
				last_departure = &last_departure2;
			}
			d_index++;//start counting events at 1, so increment first

					  //create a new departure event
			eo = create_event(d, d_index, 0, *next_departure);

			(*last_departure) = (*next_departure);//we need this time to determine when we start processing
												  //the next packet
			(*next_departure) = 0;
			if (d_index < a_index) {//we have packets in the queue
				(*next_departure) = (*last_departure) + (1 / departures.front());//get next departure time
				departures.pop();
			}
		}
		else if (next_arrival != 0 && (next_arrival < next_observation || next_observation == 0)) {
			a_index++;
			eo = create_event(a, a_index, 0, next_arrival);
			if (next_departure1 == 0) {//if the system was empty before this arrival, this is the next
									   //parcket to be serviced
				next_departure1 = next_arrival + (1 / departures.front());
				departures.pop();
			}
			else if (next_departure2 == 0) {//if the system was empty before this arrival, this is the next
											//parcket to be serviced
				next_departure2 = next_arrival + (1 / departures.front());
				departures.pop();
			}
			next_arrival = 0;
		}
		else {
			o_index++;
			eo = create_event(o, o_index, 0, next_observation);
			next_observation = 0;
		}
		ad.push(eo);
	}
	return 0;
}

//end of creating queues

//actions for event processing

void arrive(unsigned long long int & num_arrivals, event_occurence &curr_arr, unsigned long long int num_departures,
	int num_servers, unsigned long long int & num_in_queue, unsigned long long int &num_arrival_events) {
	/*
	*perform actions upon packet arrival events
	*/
	num_arrival_events++;
	unsigned long long int curr_in_queue;
	if (curr_arr.e.skip == 0) {
		num_arrivals++;
	}
	curr_in_queue = num_arrivals - num_departures;
    //per TA's announcement N_A and N_O are the number of packets in the system not queue
	num_in_queue += curr_in_queue - num_servers;
	assert(num_in_queue <= ULLONG_MAX);
}

void depart(unsigned long long int &num_departures, event_occurence &curr) {
	/*
	*perform actions upon packet departure events
	*/
	num_departures++;
}

void observe(unsigned long long int num_arrivals, unsigned long long int num_departed, unsigned long long int &num_empty,
	unsigned long long int & num_observations, unsigned long long int &num_buffer, int num_servers) {
	/*
	*perform actions upon observation events
	*/
	num_observations++;
	unsigned long long int curr_buffer = num_arrivals - num_departed;
	if (curr_buffer == 0) {
		num_empty++;
	}
	
	num_buffer += curr_buffer;
	assert(num_buffer <= ULLONG_MAX);
}

//end of event processing

//simulation

void simulate_queue(double &T, double &lambda, double &L1, double &L2, double &alpha, double &C, int K,
	double &mu, double &p, int num_servers, double &PIdle, double &Ploss, double &N_a, double &N_o,
	int(*arrival_dist)(double &, std::queue<double> &, double &, unsigned),
	void(*departure_dist)(int, std::queue<double>&, double &, double &, double &, double &, double &, unsigned),
	unsigned service_seed, unsigned arrival_seed, unsigned obs_seed) {
	/*
	runs the desired simulation based on the input functions and parameters
	*/
	//declare queues and generate events
	std::queue<double> arrivals;
	int num_arrivals = arrival_dist(T, arrivals, lambda, arrival_seed);
	std::queue<double> departures;
	departure_dist(num_arrivals, departures, C, p, mu, L1, L2, service_seed);
	std::queue<double> observations;
	generate_poisson(T, observations, alpha, obs_seed);

	std::queue<event_occurence>event_queue;
	
	//create queues
	if (num_servers == 1) {
		Ploss = arrival_departures_1_server(arrivals, departures, C, K, event_queue, observations);
	}
	else {
		Ploss = arrival_departures_2_server_infinite_buffer(arrivals, departures, C, event_queue, observations);
	}

	//we are now set up to dequeu the events and calculate the desired parameters
	unsigned long long int num_arrivals1 = 0;
	unsigned long long int num_arrival_events = 0;
	unsigned long long int num_departures = 0;
	unsigned long long int num_empty = 0;
	unsigned long long int num_observations = 0;
	unsigned long long int num_in_queue_per_arrival = 0;
	unsigned long long int num_buffer_per_observation = 0;

	//iterate through the event queue and perform correct action
	while (!event_queue.empty()) {
		event_occurence e = event_queue.front();
		event_queue.pop();
		if (e.t > T)
			break;
		switch (e.e.t) {
		case a: arrive(num_arrivals1, e, num_departures, num_servers, num_in_queue_per_arrival, num_arrival_events);
			break;
		case d: depart(num_departures, e);
			break;
		case o: observe(num_arrivals1, num_departures, num_empty, num_observations, num_buffer_per_observation, num_servers);
			break;
		default:
			std::cout << "ERROR DEFUALT" << std::endl;
			assert(false);
			break;
		}
	}
	
	PIdle = (long double)num_empty / (long double)num_observations;
	N_a = (long double)num_in_queue_per_arrival / (long double)num_arrival_events;
	N_o = (long double)num_buffer_per_observation / (long double)num_observations;
}

double get_T(double &lambda, double &L1, double &L2, double &alpha, double &C, int K, double &mu,
	double &p, int num_servers, double &PIdle, double &Ploss, double &N_a, double &N_o,
	int(*arrival_dist)(double &, std::queue<double> &, double &, unsigned),
	void(*departure_dist)(int, std::queue<double> &, double &, double &, double&, double &, double &, unsigned)) {
	/*
	Finds the value of T for which we are in steady state. also gets the results of simulation
	*/
	unsigned service_seed = std::chrono::system_clock::now().time_since_epoch().count();
	double N_a_prev = 0;
	double N_o_prev = 0;
	double Ploss_prev = 0;
	double PIdle_prev = 0;
	double T = 10000;
	double nT = T + T;
	unsigned arrival_seed = std::chrono::system_clock::now().time_since_epoch().count() + 20;
	unsigned obs_seed = std::chrono::system_clock::now().time_since_epoch().count() + 30;

	simulate_queue(T, lambda, L1, L2, alpha, C, K, mu, p, num_servers, PIdle_prev, Ploss_prev,
		N_a_prev, N_o_prev, arrival_dist, departure_dist, service_seed, arrival_seed, obs_seed);
	simulate_queue(nT, lambda, L1, L2, alpha, C, K, mu, p, num_servers, PIdle, Ploss,
		N_a, N_o, arrival_dist, departure_dist, service_seed, arrival_seed, obs_seed);

	double N_a_diff = std::abs((N_a - N_a_prev) / N_a);
	double N_o_diff = std::abs((N_o - N_o_prev) / N_o);
	double PIdle_diff = std::abs((PIdle - PIdle_prev) / PIdle);
	double Ploss_diff = std::abs((Ploss - Ploss_prev) / Ploss);

	while (!((N_a_diff <= 0.05 || std::isnan(N_a_diff)) && (N_o_diff <= 0.05 || std::isnan(N_o_diff)) && (PIdle_diff <= 0.05 || std::isnan(PIdle_diff)) && (Ploss_diff <= 0.05 || std::isnan(Ploss_diff)))) {
		N_a_prev = N_a;
		N_o_prev = N_o;
		Ploss_prev = Ploss;
		PIdle_prev = PIdle;
		nT += T;
		simulate_queue(nT, lambda, L1, L2, alpha, C, K, mu, p, num_servers, PIdle, Ploss,
			N_a, N_o, arrival_dist, departure_dist, service_seed, arrival_seed, obs_seed);
		N_a_diff = std::abs((N_a - N_a_prev) / N_a);
		N_o_diff = std::abs((N_o - N_o_prev) / N_o);
		PIdle_diff = std::abs((PIdle - PIdle_prev) / PIdle);
		Ploss_diff = std::abs((Ploss - Ploss_prev) / Ploss);
	}
	return nT;
}

void sim_multi_param(std::string & fname, double & start, double &end, double &step, double &L1, double &L2, double &C, int K,
	double &p, int num_servers, int(*arrival_dist)(double &, std::queue<double> &, double &, unsigned),
	void(*departure_dist)(int, std::queue<double> &, double &, double &, double&, double &, double &, unsigned)) {
	/*
	iterates rho between start and end with a step size of step and simmulates the specified queue.
	stores the results in an input csv file.
	*/
	std::ofstream results;
	results.open(fname);
	results << "rho,T,lambda,alpha,C,K,p,num_servers,E[N] according to the arrival process,E[N]";
	results << " according to the observer process,PIdle,PLoss" << std::endl;
	results.close();
	for (double rho = start; rho <= end; rho += step) {
		double lambda = num_servers*rho*C / (p*L1 + (1 - p)*L2);
		double mu = lambda / (rho*num_servers);
		double alpha = lambda *0.9;
		double PIdle, Ploss, N_a, N_o;
		double T = get_T(lambda, L1, L2, alpha, C, K, mu, p, num_servers,
			PIdle, Ploss, N_a, N_o, arrival_dist, departure_dist);
		results.open(fname, std::ios_base::app);
		results << rho << "," << T << "," << lambda << "," << alpha << "," << C << "," << K << "," << p << ",";
		results << num_servers << "," << N_a << "," << N_o << "," << PIdle << "," << Ploss << std::endl;
		results.close();
	}
}

int main()
{
	double T; double lambda; double L = 20000; double L1 = 16000; double L2 = 0; double alpha = 0.0315; int K = -1; double p = 1;
	T = 10000;
	lambda = 0.035;
	double C = 2000000;
	double unused = 0;



	std::string fname;
	double start = 0.35; double end = 0.96; double step = 0.05;
	
	//q3
	fname = "M_M_1_inf.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_exponential);
	start = 0.35;
	fname = "D_M_1_inf_rho.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_deterministic_arrival, &generate_exponential);
	L2 = 21000;
	p = 0.2;
	fname = "M_G_1_inf.csv";
	sim_multi_param(fname, start, end, step, L1, L2, C, K, p, 1, &generate_poisson, &generate_bipolar);
	//q4
    std::cout<<"q4"<<std::endl;
	start = 1.5;
	end = 1.6;
	step = 1;
	fname = "D_M_1_inf_rho_big.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_deterministic_arrival, &generate_exponential);
	p = 1;
	//q8
    std::cout<<"q8"<<std::endl;
	L2 = 0;
	start = 0.4;
	end = 2.01;
	step = 0.1;
	K = 10;
	fname = "M_D_1_K10_rho_small.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_deterministic_service);
	K = 50;
	fname = "M_D_1_K50_rho_small.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_deterministic_service);
	K = 100;
	fname = "M_D_1_K100_rho_small.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_deterministic_service);

	start = 2;
	end = 3.1;
	step = 0.2;
	K = 10;
	fname = "M_D_1_K10_rho_big.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_deterministic_service);
	K = 50;
	fname = "M_D_1_K50_rho_big.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_deterministic_service);
	K = 100;
	fname = "M_D_1_K100_rho_big.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 1, &generate_poisson, &generate_deterministic_service);
	
	//q9
    std::cout<<"q9"<<std::endl;
	C = 1000000;
	K = -1;
	start = 0.35;
	end = 0.96;
	step = 0.05;
	fname = "M_D_2_inf.csv";
	sim_multi_param(fname, start, end, step, L, unused, C, K, p, 2, &generate_poisson, &generate_deterministic_service);

}
