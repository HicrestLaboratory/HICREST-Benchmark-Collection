/* 
   Copyright (c) 2016, 2018 Andreas F. Borchert
   All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
   KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*
   fused-random-chase: like random-chase but chases n independent random
   pointer chains simultaneously (fused) for n from 1 to 8.

   Each chain is a distinct random cyclic permutation over a buffer of
   the given size, so all fuse factors share the same memory footprint
   per chain. The fused loop body advances every chain by one hop in
   strict sequence, keeping the dependency chains serialized within each
   stream while allowing the CPU to overlap misses across streams.

   Output is a table analogous to fused-linear-chase: rows are buffer
   sizes (same size progression as random-chase), columns are fuse
   factors 1..8. Each cell reports the aggregate data-access speed in
   GiB/s.

   Preprocessor knobs (same as random-chase):
     MIN_SIZE    – smallest buffer in bytes  (default: 1024)
     MAX_SIZE    – largest  buffer in bytes  (default: 128 MiB)
     GRANULARITY – controls intermediate sizes between powers of two
                   (default: 1; for n>0 gives 2^(n-1) extra sizes per octave)
*/

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <printf.hpp>          /* https://github.com/afborchert/fmt */
#include "walltime.hpp"
#include "uniform-int-distribution.hpp"

/* ------------------------------------------------------------------ */
/* Random cyclic chain construction (identical to random-chase.cpp)   */
/* ------------------------------------------------------------------ */

static void** create_random_chain(std::size_t size) {
   std::size_t len = size / sizeof(void*);
   void** memory = new void*[len];
   UniformIntDistribution uniform;

   std::size_t* indices = new std::size_t[len];
   for (std::size_t i = 0; i < len; ++i) indices[i] = i;
   for (std::size_t i = 0; i < len - 1; ++i) {
      std::size_t j = i + uniform.draw(len - i);
      if (i != j) std::swap(indices[i], indices[j]);
   }
   for (std::size_t i = 1; i < len; ++i)
      memory[indices[i-1]] = (void*) &memory[indices[i]];
   memory[indices[len-1]] = (void*) &memory[indices[0]];

   delete[] indices;
   return memory;
}

/* ------------------------------------------------------------------ */
/* Fused chasing infrastructure (mirrors fused-linear-chase.cpp)      */
/* ------------------------------------------------------------------ */

template<typename Body, typename Object>
inline void fused_action(Body body, Object& object) {
   body(object);
}
template<typename Body, typename Object, typename... Objects>
inline void fused_action(Body body, Object& object, Objects&... objects) {
   body(object);
   fused_action(body, objects...);
}

/* must not be static – prevents the optimizer from eliding the chase */
volatile void* fused_random_global;

template<typename... Pointers>
double fused_chase(std::size_t count, Pointers&... ptrs) {
   WallTime<double> walltime;
   while (count-- > 0)
      fused_action([](void**& p){ p = (void**) *p; }, ptrs...);
   auto elapsed = walltime.elapsed();
   fused_action([](void**& p){ fused_random_global = *p; }, ptrs...);
   return elapsed;
}

/* ------------------------------------------------------------------ */
/* log2 helper (same as random-chase.cpp)                             */
/* ------------------------------------------------------------------ */

static unsigned int log2(std::size_t val) {
   unsigned int count = 0;
   while (val >>= 1) ++count;
   return count;
}

/* ------------------------------------------------------------------ */
/* Configuration macros                                               */
/* ------------------------------------------------------------------ */

#ifndef MIN_SIZE
#  define MIN_SIZE 1024
#endif
#ifndef MAX_SIZE
#  define MAX_SIZE (1024u * 1024u * 128u)
#endif
#ifndef GRANULARITY
#  define GRANULARITY (1u)
#endif

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main() {
   /* Print header (same style as fused-linear-chase) */
   fmt::printf("                                          "
               "data access speeds in GiB/s\n");
   fmt::printf("      fuse");
   for (int i = 1; i <= 8; ++i) fmt::printf("%12d", i);
   fmt::printf("\n   memsize\n");

   for (std::size_t memsize = MIN_SIZE; memsize <= MAX_SIZE;
        memsize += (std::size_t{1} <<
           (std::max(GRANULARITY, log2(memsize)) - GRANULARITY))) {

      /* Number of hops per chain – same formula as random-chase */
      std::size_t count = std::max(memsize * 16, std::size_t{1} << 30);

      fmt::printf(" %9u", memsize);

      /* Allocate 8 independent random chains of the same size */
      void *m1, *m2, *m3, *m4, *m5, *m6, *m7, *m8;
      fused_action([=](void*& p){
         p = create_random_chain(memsize);
      }, m1, m2, m3, m4, m5, m6, m7, m8);

      /* Working pointers – reset before each fuse measurement so
         every fuse-k run starts from the beginning of each chain */
      void **p1 = (void**)m1, **p2 = (void**)m2,
           **p3 = (void**)m3, **p4 = (void**)m4,
           **p5 = (void**)m5, **p6 = (void**)m6,
           **p7 = (void**)m7, **p8 = (void**)m8;

      auto print_result = [=](int fuse, double t) {
         /* volume = bytes traversed across all chains */
         double volume = static_cast<double>(sizeof(void*)) * count * fuse;
         double speed  = volume / t / (1u << 30); /* GiB/s */
         fmt::printf("  %10.5lf", speed);
         std::cout.flush();
      };

      /* Reset working pointers before each measurement so the fuse-k
         run always chases from a defined starting position. The chains
         themselves are not recreated (that would be prohibitively slow
         for large buffers), so successive runs share the same random
         permutation – which is fine for timing purposes. */
      p1=(void**)m1; print_result(1, fused_chase(count, p1));
      p1=(void**)m1; p2=(void**)m2;
         print_result(2, fused_chase(count, p1, p2));
      p1=(void**)m1; p2=(void**)m2; p3=(void**)m3;
         print_result(3, fused_chase(count, p1, p2, p3));
      p1=(void**)m1; p2=(void**)m2; p3=(void**)m3; p4=(void**)m4;
         print_result(4, fused_chase(count, p1, p2, p3, p4));
      p1=(void**)m1; p2=(void**)m2; p3=(void**)m3; p4=(void**)m4;
      p5=(void**)m5;
         print_result(5, fused_chase(count, p1, p2, p3, p4, p5));
      p1=(void**)m1; p2=(void**)m2; p3=(void**)m3; p4=(void**)m4;
      p5=(void**)m5; p6=(void**)m6;
         print_result(6, fused_chase(count, p1, p2, p3, p4, p5, p6));
      p1=(void**)m1; p2=(void**)m2; p3=(void**)m3; p4=(void**)m4;
      p5=(void**)m5; p6=(void**)m6; p7=(void**)m7;
         print_result(7, fused_chase(count, p1, p2, p3, p4, p5, p6, p7));
      p1=(void**)m1; p2=(void**)m2; p3=(void**)m3; p4=(void**)m4;
      p5=(void**)m5; p6=(void**)m6; p7=(void**)m7; p8=(void**)m8;
         print_result(8, fused_chase(count, p1, p2, p3, p4, p5, p6, p7, p8));

      /* Free all chains */
      fused_action([](void*& p){
         delete[] (void**) p;
      }, m1, m2, m3, m4, m5, m6, m7, m8);

      fmt::printf("\n"); std::cout.flush();
   }
}
