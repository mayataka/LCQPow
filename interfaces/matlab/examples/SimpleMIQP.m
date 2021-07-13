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
nv = 1 + nbits + 1;
nc = 1;

% Variables and box constraints
w = SX.sym('w', nv, 1);
x = w(1);
b = w(2:nv-1);
one = w(nv);

% Box Constraints
lb = zeros(nv,1);
ub = inf(nv,1);
lb(nv) = 1;
ub(nv) = 1;

% Objective
a = 10*pi;
obj = (x - a)^2;

% Constraints
constr = {x - 2.^(0:(nbits-1))*b};
    
lbA = [0]; 
ubA = [0];

% Complementarities
compl_L = {b};
compl_R = {one - b};
    
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
%problem.Q = problem.Q + eps*eye(nv);

% Solve LCQP
params.printLevel = 2;
%params.x0 = [10*pi; 0.5*ones(nv-2,1); 1];
params.initialPenaltyParameter = eps;
params.stationarityTolerance = 1e-4;
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

x(1) = round(10*pi);
problem.obj(x)