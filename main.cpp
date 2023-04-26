#include <dolfin.h>
#include "VlasovPoisson.h"
#include "Poisson.h"
#include "SUPGSolver.h"



using namespace dolfin;


int main()
{
    
    parameters["reorder_dofs_serial"] = false;
    set_log_active(false);
    std::size_t N = 32;
    
    Point p1(xmin,vmin);
    Point p2(xmax,vmax);
    
    auto mesh_spatial = std::make_shared<IntervalMesh>(N,xmin,xmax);
    auto mesh_phase = std::make_shared<RectangleMesh>(p1,p2,N,N);
    
    auto pbc = std::make_shared<PeriodicBoundaryX>();

    auto V_spatial = std::make_shared<Poisson::Form_a_FunctionSpace_0>(mesh_spatial,pbc);
    auto V_phase = std::make_shared<SUPGSolver::Form_a_vp::TestSpace>(mesh_phase,pbc);
    auto V_phase_DG = std::make_shared<SUPGSolver::Form_a_tau::TestSpace>(mesh_phase);
   
    Indata ini;
    auto f = std::make_shared<Function>(V_phase);
    f->interpolate(ini);
    auto velo = std::make_shared<FieldV>();
    auto zero = std::make_shared<Zero>();

    auto Phi = std::make_shared<Function>(V_phase);
    auto rho = std::make_shared<Function>(V_spatial->sub(0)->collapse());
    auto phi = std::make_shared<Function>(V_spatial->sub(0)->collapse());
    auto Mphi = std::make_shared<Function>(V_spatial);
    auto tau = std::make_shared<Function>(V_phase_DG);
    
    auto a_vp = std::make_shared<SUPGSolver::Form_a_vp>(V_phase, V_phase);
    auto L_vp = std::make_shared<SUPGSolver::Form_L_vp>(V_phase);
    auto a_tau = std::make_shared<SUPGSolver::Form_a_tau>(V_phase_DG, V_phase_DG);
    auto L_tau = std::make_shared<SUPGSolver::Form_L_tau>(V_phase_DG);
    auto a_poisson = std::make_shared<Poisson::Form_a>(V_spatial, V_spatial);
    auto L_poisson = std::make_shared<Poisson::Form_L>(V_spatial);
    auto E = std::make_shared<Poisson::Form_E>(mesh_spatial);

    a_vp->v = velo;
    L_vp->v = velo;
    L_tau->v = velo;
    
    std::vector<double> cor_spatial = rho->function_space()->tabulate_dof_coordinates();
    std::vector<double> cor_phase = V_phase->tabulate_dof_coordinates();
    
    std::size_t node_phase = V_phase->dim();
    std::size_t node_spatial = rho->function_space()->dim();
    std::size_t node_velocity = node_phase/(node_spatial-1);
    //return 0;
    
    auto ero = std::make_shared<Constant>(0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    DirichletBC bc(V_phase, ero, boundary);
    
    std::vector<std::vector<double>> map(node_spatial,std::vector<double>(node_phase,0));
    
    for(std::size_t i=0;i<node_spatial;i++)
    {
        double x_co = cor_spatial[i];
        if(x_co == 0)
        {
            x_co = xmax;
        }
        for(std::size_t j=0;j<node_phase;j++)
        {
            if(cor_phase[2*j]==x_co)
            {
                if(cor_phase[2*j+1]==vmax or cor_phase[2*j+1]==vmin)
                {
                    map[i][j] = 0.5;
                }
                else
                {
                    map[i][j] = 1;
                }
            }
        }
    }

    double t = 0;
    double T = 45;
    double tstep = 0.125;
    auto step = std::make_shared<Constant>(tstep);
    auto hstep = std::make_shared<Constant>(tstep/2);
    //a_vp->k = step;
    //L_vp->k = step;
    
    std::vector<double> f_arr(node_phase,0);
    std::vector<double> rho_arr(node_spatial,0);
    
    File file("results/temperature.pvd");
    file << std::pair<const Function*, double>(f.get(), t);
    
    a_vp->tau = zero;
    L_vp->tau = zero;

    double L = vmax-vmin;
    int kk = 0;
    while (t <= T ) //- DOLFIN_EPS_LARGE)
    {
    
        f->vector()->get_local(f_arr);
        a_vp->k = hstep;
        a_vp->phi = zero;
        L_vp->phi = zero;
        L_vp->u0 = f;
        
        solve(*a_vp == *L_vp, *f, bc);
        
        for(std::size_t i=0;i<node_spatial;i++)
        {
            rho_arr[i] = std::inner_product(f_arr.begin(), f_arr.end(), map[i].begin(), 0.0)*L/(node_velocity-1)-1;
        }
        rho->vector()->set_local(rho_arr);
        
        L_poisson->f = rho;
        solve(*a_poisson == *L_poisson, *Mphi);
        
        phi->interpolate((*Mphi)[0]);
        Phi->interpolate((*Mphi)[0]);
        E->phi = phi;
        std::cout<<sqrt(assemble(*E))<<"\n";
    

        a_vp->k = step;
        a_vp->phi = Phi;
        L_vp->phi = Phi;
        a_vp->v = zero;
        L_vp->v = zero;
        L_vp->u0 = f;
        solve(*a_vp == *L_vp, *f, bc);
        
        a_vp->k = hstep;
        a_vp->phi = zero;
        L_vp->phi = zero;
        a_vp->v = velo;
        L_vp->v = velo;
        L_vp->u0 = f;
        solve(*a_vp == *L_vp, *f, bc);
        
    
        
        kk++;
        t += tstep;
        
        if(kk % 5 == 0)
        {
            file << std::pair<const Function*, double>(f.get(), t);
        }
        
    }
    
    

    return 0;
}


