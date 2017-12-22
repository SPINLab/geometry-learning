/*
Source: http://www3.cs.stonybrook.edu/~algorith/implement/turn/distrib/sim.c
*/

/* Date: Sun, 21 Jun 92 19:33:02 -0400
From: ressler@cs.cornell.edu (Gene Ressler)
Message-Id: <9206212332.AA03090@durin.cs.cornell.edu>
Received: by durin.cs.cornell.edu (5.65/N-0.12)
	id AA03090; Sun, 21 Jun 92 19:32:59 -0400
To: jsbm@cs.cornell.edu
Subject: Code for polygon similarity..
Status: RO

Hi.

I remembered you're interested in the implementation
of the algorithm in the `paper of many authors', but didn't
have an e-mail address for you until Bruce just sent it.
The code follows.

Please note how a heap (priority queue) is used to improve
the run time given in the paper just a bit.

The end result is that it does 21 of the matches from the 
paper in about 175 milliseconds (about 8ms apiece) on
a Sparc 1+.  I include the data file.

I'd be most interested in learning about bugs, as the code
should be robust --- there is a rough proof correctness
that the floating point math can't mess up the combinatorics.

Gene

----- sim.c -----
*/

/*
 *  Implementation of "An Efficiently Computable Metric
 *  for Comparing Polygonal Shapes," by Arkin, Chew, Huttenlocher,
 *  Kedem, and Mitchel (undated).  This expands a little on the
 *  cited reference to achieve O(n) space and O(mn log n)
 *  run time.
 *
 *  This could be improved to O(min m,n) and O(mn log min m,n)
 *  by selecting the smallest of the 2 polys to create the initial
 *  event heap.  See init_events().
 *
 *  Variable names match the article.
 *
 *  Implementation (c) Eugene K. Ressler 91, 92  This source may be
 *  freely distributed and used for non-commercial purposes, so long 
 *  as this comment is attached to any code copied or derived from it.
 *
 * Make gcc version 2.1:
 *  gcc [-g] [-O2 -finline-functions] [-DCPU_TIME] sim.c -lm -o sim   
 * Turbo C 2.0 or TC++ 1.0:
 *  tcc [-O -Z -G] sim.c -o sim.exe
 *
 * Usage:
 *  sim [-n] [-p] [-r#] [< infile] [> outfile]
 *
 * Options:
 *  -n suppresses brute force updates to reduce numerical error.
 *     This typically speeds things up a lot for big polygons.
 *     I haven't seen any inputs where the updates do much good.
 *  -p prints the metric value in full double precision.
 *  -r repeats the calculation for each polygon # times to assist in
 *     performance measure for small polygons.  This number is factored
 *     out of run time reports when compiled with -DCPU_TIME, so higher
 *     numbers compensate for grainy clocks.
 *
 * I/O:
 *  A model polygon and then any number of polygons to be compared
 *  to the model are read from standard input.  For each of the
 *  latter, a single line of metric information is written to the
 *  standard output.  Polygons are expected as counterclockwise
 *  vertex lists, one vertex (x, y) per line.  Any line that does
 *  not begin with <digit>| . | - is not scanned and
 *  taken as the end of a polygon. Output lines look like:
 *
 *  <metric> <theta* (degrees)> <f corner> <g corner> <ht0 err> <slope err> [ <cpu time in ms> ]
 *
 *  Metric and theta* are as given in the paper.  The corners are
 *  vertex numbers (0-based) of the vertices at the critical
 *  event that gave the best match.  Polygon f is the last read;
 *  g is the model.  The error terms are the maximum magnitudes of
 *  errors discovered during brute force updates.  These will be zero
 *  if the -n option is invoked.
 *
 * Bugs:
 *  The input routine uses gets() with an input buffer of 80 chars,
 *  so inputs containing long lines will cause unpredictable results.
 *
 *  We should really size things dynamically.  Currently the number of
 *  points in input polys is limited by a compile-time constant.
 *
 *  The code may not work with redundant vertices or
 *  edges that fold back on their predecessors.
 *
 *  The proof of correctness for this calculation is based on IEEE
 *  floating point numbers.  All bets are off with other standards.
 *  If the proof is wrong, it will almost inevitably show up as large
 *  numbers (over, say 10^-3) in the error terms of the heuristic.
 *
 * Compiling with -DCPU_TIME causes the time for each metric calculation
 * in ms to be appended.  Unfortunately, most systems provide this accurate
 * to 1/60 sec (16.6.. ms) only.  Small polygons take much less time than
 * this to compare.  The -r option compensates for this with a hack.
 */
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#ifdef CPU_TIME
#include <sys/times.h>
#include <sys/param.h>
#endif

/*
 * Largest polygon to be handled.
 */
#define MAX_PTS 100

/*
 * Types.
 */
typedef double COORD;       /* Cartesian coordinate */

typedef struct pt {         /* Cartesian point */
    COORD x, y;
} POINT;

typedef struct poly {       /* polygon of n points */
    int n;
    POINT pt[MAX_PTS];
} POLY_REC, *POLY;

typedef struct leg {        /* single leg of a turning rep polygon */
    double theta;           /* heading of the leg */
    double len;             /* length in original coordinates */
    double s;               /* cumulative arc length in [0,1] of start */
} LEG;

typedef struct turnrep {    /* polygon in turning rep */
    int n;
    double total_len;
    LEG leg[MAX_PTS];
} TURN_REP_REC, *TURN_REP;

typedef struct event {      /* critical event */
    double t;               /* "f shift" parameter of the event */
    int fi, gi;             /* pointers into turn reps f and g */
}   EVENT_REC, *EVENT,
    HEAP[MAX_PTS+1];

/*
 * These are to make a turning rep look infinite.
 * Given the rep and the "index" of a discontinuity,
 * 0,1,..., these return s and theta adjusted by
 * the correct number of turns around the polygon.
 */
#define tr_n(tr)        tr->n
#define tr_i(tr, i)     (i % tr->n)
#define tr_len(tr, i)   tr->leg[tr_i(tr, i)].len

double tr_s(TURN_REP tr, int i)
{
    return(tr->leg[i % tr->n].s + i / tr->n);
}

double tr_smt(TURN_REP tr, int i, double t)
{
    return((tr->leg[i % tr->n].s - t) + i / tr->n);
}

double tr_theta(TURN_REP tr, int i)
{
    return(tr->leg[i % tr->n].theta + i / tr->n * 2 * M_PI);
}

/*
 * Square a double.
 */
double sqr(double x)
{
    return(x * x);
}

/*
 * Compute min of two ints.
 */
int min(int x, int y)
{
    return(x < y ? x : y);
}

/*
 * Compute floor(log_base2(x))
 */
int ilog2(int x)
{
    int l;

    l = -1;
    while (x != 0) {
        x >>= 1;
        ++l;
    }
    return(l);
}

/*
 * Return angle a, adjusted by +-2kPI so that it
 * is within [base-PI, base+PI).
 */
double turn(double a, double base)
{
    while (a - base < -M_PI) a += 2 * M_PI;
    while (a - base >= M_PI) a -= 2 * M_PI;
    return(a);
}

/* Get a line from the input file. */
int read_line(char *buf, int buf_size, int *line)
{
  int ch;

  /* Set sentinel at end of buffer. */
  buf[buf_size - 1] = buf[buf_size - 2] = '\1';

  if (fgets(buf, buf_size, stdin) == NULL)
    return 0;

  /* Count successfully retrieved line. */
  ++*line;

  /* Purge rest of line if it was longer than buffer. */
  if (buf[buf_size - 1] == '\0' && buf[buf_size - 2] != '\n') {
    do { 
      ch = getchar();
    } while (ch != EOF && ch != '\n');
    fprintf(stderr, "warning: truncated line %d\n", *line);
  }
  return 1;
}

/*
 * Read a polygon from the standard input.  One (x,y) pair
 * is expected per line.  EOF or a line not starting with
 * a digit, -, or . end the polygon.  Returns 1 if a valid
 * polygon (3 or more vertics) was read.  Unparsable input
 * and polygons too big or small are reported.  An abort
 * ensues in all cases.
 */
int read_poly(POLY poly)
{
    static int line = 0;
    int i;
    char buf[82];

    i = 0;
    while(read_line(buf, sizeof buf, &line) &&
        (isdigit(buf[0]) || buf[0] == '.' || buf[0] == '-')) {
        if (sscanf(buf, "%lf %lf", &poly->pt[i].x, &poly->pt[i].y) != 2) {
            fprintf(stderr, "line %d: bad point\n", line);
            exit(1);
        }
        if (++i >= MAX_PTS) {
            fprintf(stderr, "line %d: polygon too big\n", line);
            exit(1);
        }
    }
    if (i > 0 && i < 3) {
        fprintf(stderr, "line %d: null polygon\n", line);
        exit(1);
    }
    return(poly->n = i);
}

/*
 * Convert a polygon to a turning rep.  This computes the
 * absolute angle of each leg wrt the x-axis, then adjusts
 * this to within PI of the last leg to form the turning
 * angle.  Finally, the total length of all legs is used
 * to compute the cumulative normalized arc length of each
 * discontinuity, s.
 */
void poly_to_turn_rep(POLY p, TURN_REP t)
{
    int n, i0, i1;
    double dx, dy, theta1, theta0, total_len, len;

    n = t->n = p->n;
    total_len = 0;
    for (theta0 = 0, i0 = 0; i0 < n; theta0 = theta1, ++i0) {
        /*
         * Look one vertex ahead of i0 to compute the leg.
         */
        i1 = (i0 + 1) % n;
        dx = p->pt[i1].x - p->pt[i0].x;
        dy = p->pt[i1].y - p->pt[i0].y;
        t->leg[i0].theta = theta1 = turn(atan2(dy, dx), theta0);
        total_len += t->leg[i0].len = sqrt(dx * dx + dy * dy);
    }
    t->total_len = total_len;
    for (len = 0, i0 = 0; i0 < n; ++i0) {
        t->leg[i0].s = len/total_len;
        len += t->leg[i0].len;
    }
}

/*
 * Fill in a turn rep with a rotated version of an
 * original.  Normalized arc lengths start at 0 in
 * the new representation.
 */
void rotate_turn_rep(TURN_REP t, int to, TURN_REP r)
{
    int ti, ri, n;
    double len, total_len;
    LEG *l;

    n = r->n = t->n;
    total_len = r->total_len = t->total_len;
    for (ti = to, ri = 0; ri < n; ++ti, ++ri) {
        l = &r->leg[ri];
        l->theta = tr_theta(t, ti);
        l->len = tr_len(t, ti);
        l->s = tr_s(t, ti);
    }
    for (len = 0, ri = 0; ri < n; ++ri) {
        r->leg[ri].s = len/total_len;
        len += r->leg[ri].len;
    }
}

/*
 * In one O(m + n) pass over the turning reps of the polygons
 * to be matched, this computes all the terms needed to incrementally
 * compute the metric.  See the paper.
 */
void init_vals(TURN_REP f, TURN_REP g, double *ht0_rtn, double *slope_rtn, double *a_rtn)
{
    int i, n;           /* loop params */
    int fi, gi;         /* disconts that bound current strip */
    double ht0, slope;  /* per paper */
    double a;           /* alpha in the paper */
    double last_s;      /* s at left edge of current strip */
    double ds;          /* width of strip */
    double dtheta;      /* height of strip */

    last_s = 0;
    /*
     * First strip is between 0 and the
     * earliest of 1'th f and g disconts.
     */
    gi = 1; fi = 1;
    /*
     * Zero accumulators and compute initial slope.
     */
    ht0 = a = 0;
    slope = (tr_s(g, 1) < tr_s(f, 1)) ? 0 : -sqr(tr_theta(g, 0) - tr_theta(f, 0));
    /*
     * Count all the strips
     */
    for (i = 0, n = tr_n(f) + tr_n(g) - 1; i < n; ++i) {
        /*
         * Compute height of current strip.
         */
        dtheta = tr_theta(g, gi-1) - tr_theta(f, fi-1);
        /*
         * Determine flavor of discontinuity on right.
         */
        if (tr_s(f, fi) <= tr_s(g, gi)) {
            /*
             * It's f. Compute width of current strip,
             * then bump area accumulators.
             */
            ds = tr_s(f, fi) - last_s;
            a += ds * dtheta;
            ht0 += ds * dtheta * dtheta;
            /*
             * Determine flavor of next strip.  We know it's ff
             * or fg.  In latter case, bump accumulator.  Note
             * we've skipped the first strip.  It's added as the
             * "next" of the last strip.
             */
            if (tr_s(f, fi+1) > tr_s(g, gi))
                slope += sqr(tr_theta(f, fi) - tr_theta(g, gi-1));
            /*
             * Go to next f discontinuity.
             */
            last_s = tr_s(f, fi++);
        }
        else {
            /*
             * Else it's g ...
             */
            ds = tr_s(g, gi) - last_s;
            a    += ds * dtheta;
            ht0 += ds * dtheta * dtheta;
            /*
             * ... and next strip is gg or gf, and again
             * we're interested in the latter case.
             */
            if (tr_s(g, gi+1) >= tr_s(f, fi))
                slope -= sqr(tr_theta(g, gi) - tr_theta(f, fi-1));
            /*
             * Go to next g discontinuity.
             */
            last_s = tr_s(g, gi++);
        }
    }
    /*
     * Set up all return values.
     */
    *ht0_rtn = ht0; *slope_rtn = slope; *a_rtn = a;
}

/*
 * Recompute ht0 and slope for the current event.
 * Renormalize the turning reps so that the event
 * discontinuities are first in each.  This keeps
 * all s values within [0,1) while recomputing so
 * that all are represented with the same precision.
 * If we check that no other events are pending within
 * machine epsilon of t for (fi,gi) before calling this,
 * numerical stability is guaranteed (unlike the first
 * two ways I tried).
 */
void reinit_vals(TURN_REP f, TURN_REP g, int fi, int gi, double *ht0_rtn, double *slope_rtn)
{
    double a;
    TURN_REP_REC fr, gr;

    rotate_turn_rep(f, fi, &fr);
    rotate_turn_rep(g, gi, &gr);
    init_vals(&fr, &gr, ht0_rtn, slope_rtn, &a);
}

/* Compute number of events between successive reinits
   that will not blow the asymptotice complexity bound. */
int reinit_interval(TURN_REP f, TURN_REP g)
{
  return tr_n(f) * tr_n(g) / (min(tr_n(f), tr_n(g)) * ilog2(tr_n(g)));
}

/*
 * Following are routines to maintian the event heap.  This
 * is initialized with one event per g discontinuity, namely,
 * the one due to the f discontinuity closest to the right.
 * The sort key is the "f shift" parameter t.  As the algorithm
 * runs, it draws an event (of min t) from the heap, handles it,
 * then inserts the event due to the *next* f discontinuity
 * associated with the same g discontinuity (unless this event
 * would have t>1).
 *
 * The heap insert and delete are minor modifications of pseudo-code
 * from Horowitz and Sahni, Computer Algorithms.
 */
static HEAP event;
static int n_events = 0;

/*
 * Insert a new event in the heap.
 */
void add_event(TURN_REP f, TURN_REP g, int fi, int gi)
{
    double t;
    int i, j;
    EVENT e;

    if ((t = tr_s(f, fi) - tr_s(g, gi)) > 1)
        return;
    j = ++n_events;
    i = n_events/2;
    while (i > 0 && event[i].t > t) {
        event[j] = event[i];
        j = i;
        i = i/2;
    }
    e = &event[j];
    e->t = t;
    e->fi = fi;
    e->gi = gi;
}

/*
 * Remove the event of min t from the heap and return it.
 */
EVENT_REC next_event(void)
{
    int i;
    EVENT_REC next = event[1];      /* remember event to return */
    EVENT_REC e = event[n_events];  /* new root (before adjust) */

    --n_events;
    i = 2;
    while (i <= n_events) {
        if (i < n_events && event[i].t > event[i+1].t)
            ++i;
        if (e.t <= event[i].t)
            break;
        else {
            event[i/2] = event[i];
            i *= 2;
        }
    }
    event[i/2] = e;
    return(next);
}

/*
 * Peek at the next t to come off the heap without
 * removing the element.
 */
#define next_t()    event[1].t

/*
 * Scan the turning reps to create the initial
 * events in the heap as described above.
 */
void init_events(TURN_REP f, TURN_REP g)
{
    int fi, gi;

    n_events = 0;
    /*
     * Cycle through all g discontinuities, including
     * the one at s = 1.  This takes care of s = 0.
     */
    for (fi = gi = 1; gi <= tr_n(g); ++gi) {
        /*
         * Look for the first f discontinuity to the
         * right of this g discontinuity.
         */
        while (tr_s(f, fi) <= tr_s(g, gi))
            ++fi;
        add_event(f, g, fi, gi);
    }
}

/*
 * The heart of the algorithm:  Compute the minimum value of the
 * integral term of the metric by considering all critical events.
 * This also returns the theta* and event associated with the minimum.
 */
double h_t0min(TURN_REP f, TURN_REP g,
	       double hc0, double slope, double alpha, int d_update,
               double *theta_star_rtn, EVENT e_rtn,
               double *hc0_err_rtn, double *slope_err_rtn)
{
    int left_to_update;                 /* # disconts left until update */
    double metric2, min_metric2;        /* d^2 and d^2_min thus far */
    double theta_star, min_theta_star;  /* theta* and theta*_min thus far */
    double last_t;                      /* t of last iteration */
    double hc0_err, slope_err;          /* error mags discovered on update */
    EVENT_REC e;                        /* current event */
    EVENT_REC min_e;                    /* event of d^2_min thus far */

    static EVENT_REC init_min_e = {0,0,0};  /* implicit first event */

    /*
     * At t = 0, theta_star is just alpha, and the min
     * metric2 seen so far is hc0 - min_theta_star^2.
     * Also, no error has been seen.
     */
    min_theta_star = alpha;
    min_metric2 = hc0 - min_theta_star * min_theta_star;
    min_e = init_min_e;
    last_t = hc0_err = slope_err = 0;
    /*
     * Compute successive hc_i0's by incremental update
     * at critical events as described in the paper.
     */
    left_to_update = d_update;
    while (n_events > 0) {
        e = next_event();
        hc0 += (e.t - last_t) * slope;
        theta_star = alpha - 2 * M_PI * e.t;
        metric2 = hc0 - theta_star * theta_star;
        if (metric2 < min_metric2) {
            min_metric2 = metric2;
            min_theta_star = theta_star;
            min_e = e;
        }
        /*
         * Update slope, last t, and put next event for this g
         * discontinuity in the heap.
         */
        slope += 2*(tr_theta(f, e.fi-1) - tr_theta(f, e.fi))
                  *(tr_theta(g, e.gi-1) - tr_theta(g, e.gi));
        last_t = e.t;
        add_event(f, g, e.fi+1, e.gi);
        /*
         * Re-establish hc0 and slope now and then
         * to reduce numerical error.  If d_update is 0, do nothing.
         * Note we don't update if an event is close, as this
         * causes numerical ambiguity.  The test number could be much
         * smaller, but why tempt Murphey?  We force an update on last
         * event so there's always at least one.
         */
        if (d_update && 
	    (n_events == 0 || 
            --left_to_update <= 0 && e.t - last_t > 0.001 && next_t() - e.t > 0.001)) {

            double rihc0, rislope, dhc0, dslope;

            reinit_vals(f, g, e.fi, e.gi, &rihc0, &rislope);
            dhc0 = hc0 - rihc0;
            dslope = slope - rislope;
            if (fabs(dhc0) > fabs(hc0_err))
              hc0_err = dhc0;
            if (fabs(dslope) > fabs(slope_err))
              slope_err = dslope;
            hc0 = rihc0;
            slope = rislope;
            left_to_update = d_update;
        }
    }
    /*
     * Set up return values.
     */
    *theta_star_rtn = min_theta_star;
    *e_rtn = min_e;
    *hc0_err_rtn = hc0_err;
    *slope_err_rtn = slope_err;
    return(min_metric2);
}

/*
 * Parse options, read polygons, convert to turning reps, initialize
 * everything, compute the answers, and print the answers.  Do all
 * this in a loop until all the polygons are gone.
 */
void main(int argc, char *argv[])
{
    int update_p, precise_p, n_repeats, i;
    TURN_REP_REC trf, trg;
    TURN_REP f, g;
    POLY_REC pf, pg;
    EVENT_REC e;
    double ht0, slope, alpha, theta_star, metric2, metric, ht0_err, slope_err;

#ifdef CPU_TIME
    struct tms cpu_time;
    clock_t cpu_start_time;
    int elapsed_ms;
    int total_elapsed_ms = 0;
#define start_cpu_time() \
      (times(&cpu_time), cpu_start_time = cpu_time.tms_utime)
#define get_cpu_time() \
      (times(&cpu_time), (cpu_time.tms_utime - cpu_start_time)*1000/HZ)
#endif

    precise_p = 0;
    update_p = 1;
    n_repeats = 1;
    while(--argc) {
        ++argv;
        if (argv[0][0] == '-' && argv[0][1] != '\0')
            switch(argv[0][1]) {
                case 'p':
                    precise_p = 1;
                    break;
                case 'n':
                    update_p = 0;
                    break;
		case 'r':
		    if (sscanf(&argv[0][2], "%d", &i) == 1)
		      n_repeats = i;
		    break;
                default:
                    fprintf(stderr, "sim: unknown option\n");
                    exit(1);
            }
    }
    if (read_poly(&pg)) {
        poly_to_turn_rep(&pg, &trg);
        g = &trg;
        while (read_poly(&pf)) {
            poly_to_turn_rep(&pf, &trf);
	    f = &trf;
#ifdef CPU_TIME
	    start_cpu_time();
#endif
	    /* Performance measure repeat loop. */
	    for (i = 0; i < n_repeats; ++i) {

              init_vals(f, g, &ht0, &slope, &alpha);
              init_events(f, g);
              metric2 = h_t0min(f, g, 
			        ht0, slope, alpha,
                                update_p ? reinit_interval(f, g) : 0,
                	        &theta_star, &e, &ht0_err, &slope_err);
	    }
#ifdef CPU_TIME
	    elapsed_ms = get_cpu_time();
	    total_elapsed_ms += elapsed_ms;
#endif
            /*
             * Fixups: The value of metric2 can be a tiny
             * negative number for an exact match.  Call it 0.
             * Theta_star can be over 360 or under -360 because we
             * handle t=0 events at t=1. Normalize to [-PI,PI).
             */
            metric = metric2 > 0 ? sqrt(metric2) : 0;
            printf(precise_p ? "%.18lg %.18lg %d %d %lg %lg" : "%lg %lg %d %d %lg %lg",
                   metric, turn(theta_star, 0)*180/M_PI,
                   tr_i(f, e.fi), tr_i(g, e.gi), ht0_err, slope_err);

#ifdef CPU_TIME
	    printf(" %d\n", (elapsed_ms + (n_repeats/2))/n_repeats);
#else
	    printf("\n");
#endif

        }
    }

#ifdef CPU_TIME
    printf("total user time: %d ms\n", (total_elapsed_ms + (n_repeats/2))/n_repeats);
#endif

    exit(0);
}
