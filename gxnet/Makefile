
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

TEST_PROGS = testmatmul \
		testbackward testseeds testmnist \
		testcnn testemnist

######################################################################

COMM_OBJS = common.o eval.o utils.o im2rows.o \
		optim.o context.o activation.o layer.o network.o

######################################################################

ifeq ($(eigen),1)
CFLAGS += -DENABLE_EIGEN -I/usr/local/include/eigen3/ -Wno-error
TEST_PROGS += testeigen
endif

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

testcnn: $(COMM_OBJS) testcnn.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

testemnist: $(COMM_OBJS) testemnist.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

testmatmul: common.o testmatmul.o
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

testeigen: $(COMM_OBJS) testeigen.o
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

