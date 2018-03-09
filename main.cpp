#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <math.h>
#include <assert.h>
enum event_type {a, o, d};
struct event_id{event_type t; int d_num; int skip;};
struct event_occurence{event_id e; double t;};

/////////////////////////////////////////////////////////
/////////////////////////Q1//////////////////////////////
/////////////////////////////////////////////////////////

void exp_rv(std::vector<double> &vals, double &mu_inv){
    /*
    * generate a distribution according to an exponential random variable, with parameter mu_inv
    * vals contains the distribution
    */
    //mu_inv = 1/mu
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<>dis(0,1.0);//to be used to get uniform distribution


    for (int n = 0; n<1000; ++n){
        double val1 = dis(gen);//get next value in uniform distribution
        double val = log(1-val1)*-1*mu_inv;//mu_inv might need to be divided, need to talk to prof to find out exactly what mu_inv is
        vals.push_back(val);//store exponential distribution value in vals
    }
}

int get_expoential_q1(){
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
    for (double val :vals){
        mean += val;
    }
    mean/=1000;

    //get the variance
    for (double val : vals){
        variance += pow((val - mean),2);
    }
    variance/=1000;
    std::cout<<mean<<","<<variance<<std::endl;
    return 0;
}



/////////////////////////////////////////////////////////
/////////////////////////Q1 done/////////////////////////
/////////////////////////////////////////////////////////

event_occurence create_event(event_type et, int o, int skip, double &t){
    /*creates an event occurence*/
    event_type e = et;
    event_id ei;
    ei.t=e;
    ei.d_num= o;
    ei.skip = skip;
    
    event_occurence eo;
    eo.e = ei;
    eo.t = t;//store departure time
    return eo;
}

//create distributions

int generate_poisson(double & T, std::vector<double> &times, double &mu){
    /*
    generate a poisson distribution with parameter mu. generates values until
    the event time is greater than T and stores the values in times.

    used for arrival = M and observations
    */

    double start = 0;
    int counter = 0;
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<>dis(0,1.0);
    while (start<=T){
        start +=log(1-dis(gen))*-1/mu; //interarrival time of poisson distribution is exponentially distributed
        if (start <=T){
            times.push_back(start);
            counter ++;
        }

    }
    return counter;
}

int generate_deterministic_arrival(double & T, std::vector<double> &times, double &mu){
    /*
    generate a deterministic distribution with parameter mu. generates values until
    the event time is greater than T and stores the values in times.

    used for arrival = D
    */

    double start = 0;
    int counter = 0;
    
    while (start<=T){
        start +=mu; //interarrival time is deterministically distributed
        if (start <=T){
            times.push_back(start);
            counter ++;
        }

    }
    return counter;

}

void generate_exponential(int num, std::vector<double> &lengths, double& unused1, double &mu,  double &unused2){
    /*
    generates num values according to an exponential distribution with parameter mu
    stores these values in lengths.

    To be used for exponential service time
    */
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<>dis(0,1.0);
    for (int counter=0; counter<num; counter ++){
        lengths.push_back(log(1-dis(gen))*-1/mu);
    }
}

void generate_bipolar(int num, std::vector<double> &lengths, double &p, double& L1, double &L2){
    /*
    generates num values according to a bipolar distribution with parameter p
    and lengths L1 and L2. Stores these values in lengths.

    To be used for G service
    */
    std::random_device rand;
    std::mt19937 gen(rand());
    std::uniform_real_distribution<>dis(0,1.0);
    for (int counter=0; counter<num; counter ++){
        if (dis(gen) <= (1-p)){
            lengths.push_back(L2);
        }else{
            lengths.push_back(L1);            
        }
    }
}

void generate_deterministic_service(int num, std::vector<double> &lengths, double &unused1, double &mu,  double &unused2){
    /*
    generates num values according to a deterministic distribution with parameter mu
    stores these values in lengths.

    To be used for deterministic service time
    */
    
    for (int counter=0; counter<num; counter ++){
        lengths.push_back(mu);
    }
}

//end of distribution generation functions

//create event queues

double arrival_departures_1_server_infinite_queue(std::vector<double> &arrivals, std::vector<double> &departures, 
            double &C, std::vector<event_occurence> &ad){
    /*
    uses a vector of arrivals and departures to create the departure times when
    there is 1 server and place arrivals and departures in
    order. this is an infinite buffer
    returns the number of packets lost/number of packets processed
    //should not be used, is here for purposes of 1 of questions
    */
    int a_index = 0;//how many arrivals we have added to the queue
    int d_index = 0;//which departure we are adding to the queue
    double next_departure = 0; //next departure time, 0 =>no packet is being serviced
    double last_departure = 0; //most recent departure time
    
    
    

    
    for (double arrival : arrivals){//iterate through the arrivals
        while (next_departure !=0 && next_departure <=arrival){//we have a packet being served and it should
                                                                //finish before the next arrival occurs
            d_index ++;//start counting events at 1, so increment first
            
            //create a new departure event
            event_occurence eo = create_event(d, d_index, 0, next_departure);
            
            ad.push_back(eo);

            last_departure = next_departure;//we need this time to determine when we start processing
                                        //the next packet


            
            next_departure = 0;


            
            if (d_index < a_index){//we have packets in the queue
                next_departure = last_departure + (departures[d_index]/C);//get next departure time
            }
        }

        a_index++;//everything that has departed before this packet has arrived has been recorded
        event_occurence eo = create_event(a, a_index, 0, arrival);
    
        ad.push_back(eo);//store arrival

        if (next_departure == 0){//if the system was empty before this arrival, this is the next
                                //parcket to be serviced
            next_departure = arrival + (departures[d_index]/C);//d_index is a_index-1, but a_index is 
                                                                //1_indexed while vectors are 0_indexed
        }

        
    }
    //finish all remaining packets departing
    while (d_index <a_index){
        
        d_index ++;
        event_occurence eo = create_event(d, d_index, 0, next_departure);
        ad.push_back(eo);
        
        last_departure = next_departure;
        
        

        next_departure = 0;

        if (d_index < a_index)//we have packets in the queue
            next_departure = last_departure + (departures[d_index]/C);//get next departure time
    }
    return 0;
}

void add_departure_event_finite_queue(int &d_index, int &skip_next, double &next_departure,
    std::vector<event_occurence>&ad, double & last_departure,  double &C,
    std::vector<int> &dropped, int &a_index, int &next_to_skip, std::vector<double>& departures){
    
    d_index ++;//start counting events at 1, so increment first
    
    //create a new departure event
    event_occurence eo = create_event(d, d_index, skip_next, next_departure);
    ad.push_back(eo);

    if (skip_next == 0){
        last_departure = next_departure;//we need this time to determine when we start processing
                                    //the next packet  
    }
    
    next_departure = 0;


    if (d_index<a_index){
        if (!dropped.empty() && (d_index+1) == dropped[next_to_skip]){
            next_departure = last_departure + (departures[d_index]/C);//get next departure time
            
            skip_next = 1;
            next_to_skip++;
            if (next_to_skip == dropped.size()){
                dropped.clear();
                next_to_skip = -1;
            }
        }
        else {//if (dropped.empty() || (d_index+1) != dropped[next_to_skip]){//we have packets in the queue
            next_departure = last_departure + (departures[d_index]/C);//get next departure time
            skip_next = 0;
        }
    }

}

double arrival_departures_1_server(std::vector<double> &arrivals, std::vector<double>& departures, 
            double& C, int K, std::vector<event_occurence> &ad){
    /*
    uses a vector of arrivals and departures to create the departure times when
    there is a queue of size K and 1 server and place arrivals and departures in
    order. if K= -1, this is an infinite buffer
    returns the number of packets lost/number of packets processed
    */
    int a_index = 0;//how many arrivals we have added to the queue
    int d_index = 0;//which departure we are adding to the queue
    double next_departure = 0; //next departure time, 0 =>no packet is being serviced
    double last_departure = 0; //most recent departure time
    double num_dropped = 0;
    int skip_next = 0;
    int next_to_skip = -1;
    std::vector<int> dropped;
    
    

    
    for (double arrival : arrivals){//iterate through the arrivals
        while (next_departure !=0 && next_departure <=arrival){//we have a packet being served and it should finish before the next arrival occurs                         
            add_departure_event_finite_queue(d_index, skip_next, next_departure,
                ad, last_departure, C, dropped, a_index, next_to_skip, departures);
        }

        a_index++;//everything that has departed before this packet has arrived has been recorded
        event_occurence eo = create_event(a, a_index, 0, arrival);
        
        if (K>=0 && a_index-d_index-K-num_dropped >3)//1 for the server, 1 because we have incremented a_index, 1 because
        {//we have K spots, so it is when we have K+1 waiting that we have a problem
            num_dropped ++;
            if (dropped.empty()||next_to_skip == -1){
                next_to_skip = 0;
            }

            dropped.push_back(a_index);
            eo.e.skip = 1;
        }                   
        else{
            if (next_departure == 0){//if the system was empty before this arrival, this is the next
                                    //parcket to be serviced
                next_departure = arrival + (departures[d_index]/C);//d_index is a_index-1, but a_index is 
                                                                    //1_indexed while vectors are 0_indexed
            }
        }
        ad.push_back(eo);//store arrival
    }
    //finish all remaining packets departing

    while (d_index <a_index){
        add_departure_event_finite_queue(d_index, skip_next, next_departure,
            ad, last_departure, C, dropped, a_index, next_to_skip, departures);
    }
    return num_dropped/a_index;
}

double arrival_departures_2_server_infinite_buffer(std::vector<double>& arrivals, std::vector<double> &departures, 
            double& C, std::vector<event_occurence> &ad){
    /*
    uses a vector of arrivals and departures to create the departure times when
    there is an infinite queue and 2 servers and place arrivals and departures in
    order
    */
    int a_index = 0;//how many arrivals we have added to the queue
    int d_index = 0;//which departure we are adding to the queue
    
    double next_departure1 = 0; //next departure time, 0 =>no packet is being serviced
    double next_departure2 = 0;
    double last_departure1 = 0; //most recent departure time
    double last_departure2 = 0;
    
    double * next_departure = NULL;
    double *last_departure = NULL;

    
    for (double arrival : arrivals){//iterate through the arrivals
        while ((next_departure1 !=0 && next_departure1 <=arrival)||(next_departure2 !=0 && next_departure2 <=arrival)){//we have a packet being served and it should
                                                                //finish before the next arrival occurs
            if (next_departure1 !=0 && next_departure1 <next_departure2){
                next_departure = &next_departure1;
                last_departure = &last_departure1;
            }else{
                next_departure = &next_departure2;
                last_departure = &last_departure2;
            }
            d_index ++;//start counting events at 1, so increment first
            
            //create a new departure event
            event_occurence eo = create_event(d, d_index, 0, *next_departure);
            
            ad.push_back(eo);

            (*last_departure) = (*next_departure);//we need this time to determine when we start processing
                                            //the next packet
            (*next_departure) = 0;

            
            if (d_index <a_index)//we have packets in the queue

                (*next_departure) = (*last_departure) + (departures[d_index]/C);//get next departure time
        }

        a_index++;//everything that has departed before this packet has arrived has been recorded
        event_occurence eo = create_event(a, a_index, 0, arrival);
        
        ad.push_back(eo);//store arrival

        if (next_departure1 == 0){//if the system was empty before this arrival, this is the next
                                //parcket to be serviced
            next_departure1 = arrival + (departures[d_index]/C);//d_index is a_index-1, but a_index is 
                                                                //1_indexed while vectors are 0_indexed
        }
        else if (next_departure2 == 0){//if the system was empty before this arrival, this is the next
                                //parcket to be serviced
            next_departure2 = arrival + (departures[d_index]/C);//d_index is a_index-1, but a_index is 
                                                                //1_indexed while vectors are 0_indexed
        }

    }
    //finish all remaining packets departing
    while (d_index <a_index){
        if (next_departure1 !=0 && next_departure1 <next_departure2){
            next_departure = &next_departure1;
            last_departure = &last_departure1;
        }else{
            next_departure = &next_departure2;
            last_departure = &last_departure2;
        }
        d_index ++;
        event_occurence eo = create_event(d, d_index, 0, *next_departure);
        
        ad.push_back(eo);

        (*last_departure) = (*next_departure);
        (*next_departure) = 0;


        if (d_index <a_index)//check whether all packets finished
            (*next_departure) = (*last_departure) +departures[d_index]/C;
    }
    return 0;
}

void create_event_queue(std::vector<event_occurence> &ad, std::vector<double>& observations, 
    std::vector<event_occurence> &event_queue){
    /*
    *creates a  new event queue in event_queue. Contains the arrivals, departures and observations in order
    */
    auto ad_it = ad.begin();
    auto ob_it = observations.begin();
    auto ad_end = ad.end();
    auto ob_end = observations.end();
    int o_index = 1;
    while (ad_it != ad_end || ob_it != ob_end){
        if (ob_end == ob_it || (ad_it != ad_end && (*ad_it).t <= (*ob_it))){//determine which event is first
            event_queue.push_back(*ad_it);
            ++ad_it;
        }
        else if (ad_end == ad_it ||(ob_end != ob_it &&(*ad_it).t>(*ob_it)))//need to create a new event for observations
        {
            event_occurence eo = create_event(o, o_index, 0, *ob_it);
            
            o_index++;
            event_queue.push_back(eo);
            ++ob_it;
        }
    }

}
//end of creating queues

//actions for event processing

void arrive(long long int & num_arrivals, event_occurence &curr_arr,long long int num_departures, 
    int num_servers, long long int & num_in_queue, long long int &num_arrival_events){
    /*
    *perform actions upon packet arrival events
    */
    num_arrival_events ++;
    int curr_in_queue;
    
    if (curr_arr.e.skip == 0){
        num_arrivals ++;
    }

    curr_in_queue = num_arrivals - num_departures - num_servers;
    
    
    if (curr_in_queue < 0)
        curr_in_queue = 0;
    num_in_queue += curr_in_queue;
    assert(num_in_queue>=0);
    
    
}

void depart(long long int &num_departures, event_occurence &curr){
    /*
    *perform actions upon packet departure events
    */
    if (curr.e.skip == 1){
        return;
    }
    num_departures++;
}

void observe(long long int num_arrivals, long long int num_departed, int &num_empty, 
    long long int & num_observations, long long int &num_buffer, int num_servers){
    /*
    *perform actions upon observation events
    */
        num_observations++;
        int curr_buffer = num_arrivals - num_departed - num_servers;
        if (curr_buffer +num_servers == 0){
            num_empty ++;
        }else if (curr_buffer < 0){
            curr_buffer = 0;
        }
        
        num_buffer +=curr_buffer;
        assert(num_buffer>=0);
        //std::cout<<num_buffer<<","<<curr_buffer<<std::endl;
}

//end of event processing

//simulation

void simulate_queue(double &T, double &lambda, double &L1, double &L2, double &alpha, double &C, int K, 
        double &p, int num_servers, double &PIdle, double &Ploss, double &N_a, double &N_o,
        int (*arrival_dist)(double &, std::vector<double> &, double &), 
        void (*departure_dist)(int, std::vector<double>&, double &, double &, double &)){
        /*
        runs the desired simulation based on the input functions and parameters
        */
    double num_dropped;

    //declare queues
    std::vector<event_occurence> event_queue;
    std::vector<event_occurence>ad;
    std::vector<double> arrivals;

    //generate events
    long long int num_arrivals= arrival_dist(T, arrivals, lambda);
    std::vector<double> departures;
    departure_dist(num_arrivals, departures, p, L1, L2);
    
    //create queues
    if (num_servers == 1){
        num_dropped = arrival_departures_1_server(arrivals, departures, C, K, ad);
    }
    else{
        num_dropped = arrival_departures_2_server_infinite_buffer(arrivals, departures, C, ad);
    }
    
    
    Ploss = num_dropped/(double) num_arrivals;
    arrivals.clear();
    departures.clear();

    std::vector<double> observations;
    generate_poisson(T, observations, alpha);
    

    create_event_queue(ad, observations, event_queue);
    ad.clear();
    observations.clear();
    //we are now set up to dequeu the events and calculate the desired parameters

    long long int num_arrival_events = 0;
    num_arrivals = 0;
    long long num_departures = 0; 
    int num_empty = 0;
    long long int num_observations = 0; 
    long long int num_in_queue_per_arrival = 0;
    long long int num_buffer_per_observation = 0;

    std::ofstream results;
    results.open("queue.csv");
    results<<"time,type,id, skip"<<std::endl;
    std::ofstream r2;
    r2.open("queue2.csv");

    
    //iterate through the event queue and perform correct action
    for (event_occurence e: event_queue){
        results<<e.t<<","<<e.e.t<<","<<e.e.d_num<<","<<e.e.skip<<std::endl;
        switch (e.e.t){
            case a: arrive(num_arrivals, e, num_departures, num_servers, num_in_queue_per_arrival, num_arrival_events);
                break;
            case d: depart(num_departures, e);
                break;
            case o: observe(num_arrivals, num_departures, num_empty,  num_observations, num_buffer_per_observation, num_servers);
                break;
            default:
                std::cout<<"ERROR DEFUALT"<<std::endl;
                assert(false);
                break;
        }
        r2<<num_departures<<","<<num_arrivals<<","<<num_observations<<","<<e.e.t<<std::endl;
    }
    results.close();
    r2.close();
    PIdle = num_empty/num_observations;
    N_a = (long double)num_in_queue_per_arrival/(long double)num_arrival_events;
    N_o = (long double)num_buffer_per_observation/(long double)num_observations;
}

int get_T(double &lambda, double &L1, double &L2, double &alpha, double &C, int K, 
        double &p, int num_servers, double &PIdle, double &Ploss, double &N_a, double &N_o,
        int (*arrival_dist)(double &, std::vector<double> &, double &), 
        void (*departure_dist)(int, std::vector<double> &, double &, double &, double &)){
    /*
    Finds the value of T for which we are in steady state. also gets the results of simulation
    */
    double N_a_prev = 0;
    double N_o_prev = 0;
    double Ploss_prev = 0;
    double PIdle_prev = 0;
    double T = 1000;
    double nT = T+T;

    simulate_queue(T, lambda, L1, L2, alpha, C, K, p, num_servers, PIdle_prev, Ploss_prev, 
        N_a_prev, N_o_prev,arrival_dist, departure_dist);
    simulate_queue(nT, lambda, L1, L2, alpha, C, K, p, num_servers, PIdle, Ploss, 
        N_a, N_o,arrival_dist, departure_dist);
    
    double N_a_diff = std::abs((N_a-N_a_prev)/N_a);
    double N_o_diff = std::abs((N_o-N_o_prev)/N_o);
    double PIdle_diff = std::abs((PIdle-PIdle_prev)/PIdle);
    double Ploss_diff = std::abs((Ploss-Ploss_prev)/Ploss);
    
    while (!((N_a_diff <= 0.05 ||isnan(N_a_diff)) && (N_o_diff <= 0.05||isnan(N_o_diff)) && (PIdle_diff <= 0.05 ||  isnan(PIdle_diff)) && (Ploss_diff <= 0.05||isnan(Ploss_diff)))){
        N_a_prev = N_a;
        N_o_prev = N_o;
        Ploss_prev = Ploss;
        PIdle_prev = PIdle;
        
        nT += T;
        simulate_queue(nT, lambda, L1, L2, alpha, C, K, p, num_servers, PIdle, Ploss, 
            N_a, N_o,arrival_dist, departure_dist);
        N_a_diff = std::abs((N_a-N_a_prev)/N_a);
        N_o_diff = std::abs((N_o-N_o_prev)/N_o);
        PIdle_diff = std::abs((PIdle-PIdle_prev)/PIdle);
        Ploss_diff = std::abs((Ploss-Ploss_prev)/Ploss);
    }
    return nT;   
}

void sim_multi_param(std::string & fname,double & start, double &end, double &step, double &L1, double &L2, double &C, int K, 
        double &p, int num_servers, int (*arrival_dist)(double &, std::vector<double> &, double &), 
        void (*departure_dist)(int, std::vector<double> &, double &, double &, double &)){
    /*
    iterates rho between start and end with a step size of step and simmulates the specified queue.
    stores the results in an input csv file.
    */
    std::ofstream results;
    results.open(fname);
    results<<"rho,T,lambda,alpha,C,K,p,num_servers,E[N] according to the arrival process,E[N]";
    results<<" according to the observer process,PIdle,PLoss"<<std::endl;
    results.close();
    for (double rho = start; rho <= end; rho += step){
        double lambda = rho*C/(p*L1+(1-p)*L2);
        double alpha = lambda *0.9;
        double PIdle, Ploss, N_a, N_o;
        int T = get_T(lambda, L1, L2, alpha, C, K, p, num_servers, 
            PIdle, Ploss, N_a, N_o, arrival_dist, departure_dist);
        results.open(fname, std::ios_base::app);
        results<<rho<<","<<T<<","<<lambda<<","<<alpha<<","<<C<<","<<K<<","<<p<<",";
        results<<num_servers<<","<<N_a<<","<<N_o<<","<<PIdle<<","<<Ploss<<std::endl;
        results.close();
        N_a =0;
        N_o = 0;
        PIdle =0;
        Ploss =0;

    }
}

int main()
{
    //continue testing from simulate_queue,arrive, observe, depart
    double T; double lambda; double L1 = 10; double L2 =0; double alpha = 9; int K = 5; double p =0.5;
    T = 10000;
    lambda = 10;
    double C = 0.5;
    double unused = 0;
    double PIdle_prev;
    double Ploss_prev, N_a_prev, N_o_prev;
    std::string fname = "results.csv";
    double start = 0.35;double end = 0.95; double step = 0.05;

    
    sim_multi_param(fname, start, end, step, L1, L2, C, K, p, 1, &generate_poisson, &generate_exponential);
    
}
