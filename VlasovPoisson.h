#include <dolfin.h>

using namespace dolfin;

double const eps_ = 1.0e-6;
double const pi = DOLFIN_PI;

double const k = 0.5;
double const alpha = 0.01;

float vmax = 4.5;
float vmin = -4.5;
float xmax = 2*pi/k;
float xmin = 0;

class Indata : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = std::exp(-0.5*(x[1]-0.0)*(x[1]-0.0))*(1.0+alpha*cos(k*x[0]))/std::sqrt(2.* pi);
        //values[0] = (x[1]-0.0)*(x[1]-0.0)*std::exp(-0.5*(x[1]-0.0)*(x[1]-0.0))*(1.0+alpha*cos(k*x[0]))/std::sqrt(2.* pi);
        //values[0] = (0.9*exp(-0.5*(x[1]-0.0)*(x[1]-0.0))+0.2*exp(-2*(x[1]-4.5)*(x[1]-4.5)))*(1.0+alpha*cos(k*x[0]))/std::sqrt(2.* pi);
    }
};

class Zero : public Expression
{
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = 0;
    }
};


class PeriodicBoundaryX : public SubDomain
{
  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  { return (std::abs(x[0] - xmin) < DOLFIN_EPS); }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {
    y[0] = x[0] - xmax;
    y[1] = x[1];
  }
};

class PeriodicBoundaryXS : public SubDomain
{
  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  { return (std::abs(x[0] - xmin) <= DOLFIN_EPS); }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {
    y[0] = x[0] - xmax;
  }
};


class FieldV : public Expression
{
public:
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = x[1];
    }
};

class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return x[1] >= vmax or x[1] <= vmin;
  }
};
