
$(info    Building Core Test Program)


# location of the project source cpp and c files
SRC_C  = $(wildcard core/tests/src/*.cpp) 
SRC_C += $(wildcard core/tests/src/*.c)
SRC_C += $(wildcard core/tests/src/*/*.cpp) 
SRC_C += $(wildcard core/tests/src/*/*.c)
SRC_C += $(wildcard core/tests/src/*/*/*.cpp) 
SRC_C += $(wildcard core/tests/src/*/*/*.c)
SRC_C += $(wildcard core/tests/src/*/*/*/*.cpp) 
SRC_C += $(wildcard core/tests/src/*/*/*/*.c)

# Core repo cpp and c files
SRC_C += $(wildcard core/src/*.cpp)
SRC_C += $(wildcard core/src/*.c)
SRC_C += $(wildcard core/src/*/*.cpp)
SRC_C += $(wildcard core/src/*/*.c)
SRC_C += $(wildcard core/src/*/*/*.c)
SRC_C += $(wildcard core/src/*/*/*.cpp)
SRC_C += $(wildcard core/src/*/*/*/*.c)
SRC_C += $(wildcard core/src/*/*/*/*.cpp)


OBJ = $(addprefix $(BUILD)/, $(addsuffix .o, $(basename $(SRC_C))) )


# Core repo header files
SRC_H += $(wildcard core/include/*.h)
SRC_H += $(wildcard core/include/*/*.h)
SRC_H += $(wildcard core/include/*/*/*.h)
SRC_H += $(wildcard core/include/*/*/*/*.h)

# Vendor include directories
INC += -Ivendor/eigen

