%% Clean and Load
close all; clear all; clc;

% Load LCQPanther interface
addpath('~/LCQPow/build/lib')

% Load CasADi
addpath("~/casadi-matlab2014b-v3.5.5/");
import casadi.*;

% Load MacMPEC helpers
addpath("~/LCQPTest/MacMPECMatlab/helpers");

%% Build Problem

% Number of bits per Integer value
nbits = 17;

% Dimension
nv = 3 + 3*nbits + 1;
nc = 2 + 1;

% Variables and box constraints
w = SX.sym('w', nv, 1);
x = w(1);
y = w(2);
z = w(3);
b = w(4:nv-1);
one = w(nv);

bx = b(1:nbits);
by = b(nbits+1:2*nbits);
bz = b(2*nbits+1:3*nbits);

% Box Constraints
lb = zeros(nv,1);
ub = inf(nv,1);
lb(nv) = 1;
ub(nv) = 1;

% Objective
obj = x^2 + x*y + y^2 + y*z + z^2 + 2*x;

% Constraints
constr = {...
    x + 2*y + 3*z, ...
    x + y, ...
    x - 2.^(0:(nbits-1))*bx, ...
    y - 2.^(0:(nbits-1))*by, ...
    z - 2.^(0:(nbits-1))*bz
};
    
lbA = [4; 1; 0; 0; 0]; 
ubA = [inf; inf; 0; 0; 0];

% Complementarities
compl_L = {bx, by, bz};
compl_R = {one - bx, one - by, one - bz};
    
% Get LCQP
problem = ObtainLCQP(...
    w, ...
    obj, ...
    vertcat(constr{:}), ...
    vertcat(compl_L{:}), ...
    vertcat(compl_R{:}), ...
    lbA, ...
    ubA ...
);

% Regularization
problem.Q = problem.Q + eps*eye(nv);

% Solve LCQP
params.printLevel = 2;
x = LCQPow(...
    problem.Q, ...
    problem.g, ...
    problem.L, ...
    problem.R, ...
    problem.A, ...
    problem.lbA, ...
    problem.ubA, ...
    lb, ...
    ub, ...
    params ...
);
% Print objective at solution
problem.obj(x)