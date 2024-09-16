
CFLAGS = -std=c++17 -Wall -Werror -march=native

ifeq ($(debug),1)
CFLAGS += -g
else
CFLAGS += -O3 -DNDEBUG
endif

CPPFLAGS = $(CFLAGS)

LDFLAGS = -lstdc++ -lm

CC = gcc

######################################################################

PROGS = gxtool

TEST_PROGS = testbackward testseeds testmnist 

######################################################################

COMM_OBJS = common.o eval.o utils.o \
		optim.o context.o activation.o layer.o network.o

######################################################################

all: $(PROGS) $(TEST_PROGS)

#=====================================================================

gxtool: $(COMM_OBJS) gxtool.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

testbackward: $(COMM_OBJS) testbackward.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

testseeds: $(COMM_OBJS) testseeds.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

testmnist: $(COMM_OBJS) testmnist.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

#=====================================================================

test: $(TEST_PROGS)
	for cmd in $(TEST_PROGS) ; \
	do \
		./$$cmd; \
	done

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f *.o $(PROGS) $(TEST_PROGS) vgcore.* core

