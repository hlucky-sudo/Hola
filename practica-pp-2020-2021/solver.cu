#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "wtime.h"
#include "definitions.h"
#include "energy_struct.h"
#include "cuda_runtime.h"
#include "solver.h"

using namespace std;
__constant__ struct autodock_param_t a_params_c [MAXTYPES];

/**
* Kernel del calculo de la desolvation.
*/
__global__ void solvation (int atoms_r, int atoms_l, int nlig, float *rec_x_d, float *rec_y_d, float *rec_z_d, float *lig_x_d, float *lig_y_d, float *lig_z_d, float *ql_d,float *qr_d, float *energy_d,int *rectype, int *ligtype,int nconformations) {
	int ind1, ind2;
	float dist, temp_desolv=0, miatomo[3], e_desolv;
  float difx,dify,difz, solv_asp_1, solv_asp_2, solv_vol_1, solv_vol_2,solv_qasp_1,solv_qasp_2;
  	float  mod2x, mod2y, mod2z;	

	if(threadIdx.x < atoms_l){ 
	              ind1 = ligtype[threadIdx.x];
                miatomo[0] = *(lig_x_d + blockIdx.x * blockDim.x + threadIdx.x); ///se declaran las dimensiones
                miatomo[1] = *(lig_y_d + blockIdx.x * blockDim.x  + threadIdx.x);
                miatomo[2] = *(lig_z_d + blockIdx.x * blockDim.x  + threadIdx.x);
                solv_asp_1 = a_params_c[ind1].asp;
			          solv_vol_1 = a_params_c[ind1].vol;
                for(int j=0;j<atoms_r;j++){
                	e_desolv = 0;
                difx= (rec_x_d[j]) - miatomo[0];
        				dify= (rec_y_d[j]) - miatomo[1];
        				difz= (rec_z_d[j]) - miatomo[2];
        				mod2x=difx*difx;
        				mod2y=dify*dify;
        				mod2z=difz*difz;
        			  ind2 = rectype[j];
				        solv_asp_2 = a_params_c[ind2].asp;
				        solv_vol_2 = a_params_c[ind2].vol;
        				difx=mod2x+mod2y+mod2z;
        				dist = sqrtf(difx);
        							
        				e_desolv = ((solv_asp_1 * solv_vol_2) + (QASP * fabs(ql_d[threadIdx.x]) * solv_vol_2) + (solv_asp_2 * solv_vol_1) + (QASP * fabs(qr_d[j]) * solv_vol_1)) * exp(-difx/(2*G_D_2));
        				temp_desolv += e_desolv;	
                 
               
                }

		atomicAdd(&energy_d[blockIdx.x], temp_desolv);
        temp_desolv = 0;
	}

}





/**
* Funcion para manejar el lanzamiento de CUDA 
*/
void forces_GPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, int nconformations,struct autodock_param_t *a_params){


	cudaError_t cudaStatus; //variable para recoger estados de cuda
float dist, temp_desolv=0, e_desolv;
	//seleccionamos device
	cudaSetDevice(0); 
	//creamos memoria para los vectores para GPU _d (device)
	float *rec_x_d, *rec_y_d, *rec_z_d, *qr_d, *lig_x_d, *lig_y_d, *lig_z_d, *ql_d, *energy_d;
 int *rectype_d, *ligtype_d;

  unsigned int memsize_ligty = atoms_r * sizeof(float); 
	unsigned int memsize_recty=  atoms_r * sizeof(float); 
	unsigned int memsize_rec = atoms_r * sizeof(float);   
	unsigned int memsize_qr = atoms_r * sizeof(float);
	unsigned int memsize_lig = atoms_l * sizeof(float) * nconformations;
	unsigned int memsize_ql = atoms_l * sizeof(float);
	unsigned int memsize_energy = sizeof(float)*nconformations;


    //reservamos memoria para GPU
    cudaStatus = cudaMalloc ((void**) &rec_x_d,memsize_rec); ///Almacenamos el estado de cuda por si da un error en la creación de memoria decirlo
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 1 %d\n", cudaStatus);
 cudaStatus = cudaMalloc ((void**) &rectype_d,memsize_recty); ///Almacenamos el estado de cuda por si da un error en la creación de memoria decirlo
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 1 %d\n", cudaStatus);
 cudaStatus = cudaMalloc ((void**) &ligtype_d,memsize_ligty); ///Almacenamos el estado de cuda por si da un error en la creación de memoria decirlo
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 1 %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &rec_y_d,memsize_rec);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 2 %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &rec_z_d,memsize_rec);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 3 %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &qr_d,memsize_qr);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 4  %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &lig_x_d,memsize_lig);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 5 %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &lig_y_d,memsize_lig);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 6 %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &lig_z_d,memsize_lig);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 7 %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &ql_d,memsize_ql);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 8  %d\n", cudaStatus);

	cudaStatus = cudaMalloc ((void**) &energy_d,memsize_energy);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 9 %d\n", cudaStatus);
 

	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
		if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel 15 %d\n", cudaStatus);
   
//Ahora procederemos a pasar los datos del host al device
	if (cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al crear la memoria 11 %d\n",cudaStatus);
	cudaMemcpy(rec_x_d, rec_x, memsize_rec, cudaMemcpyHostToDevice);
	cudaMemcpy(rec_y_d, rec_y, memsize_rec, cudaMemcpyHostToDevice);
	cudaMemcpy(rec_z_d, rec_z, memsize_rec, cudaMemcpyHostToDevice);
	cudaMemcpy(qr_d, qr, memsize_qr, cudaMemcpyHostToDevice);
    cudaMemcpy(lig_x_d, lig_x, memsize_lig, cudaMemcpyHostToDevice);
	cudaMemcpy(lig_y_d, lig_y, memsize_lig, cudaMemcpyHostToDevice);
    cudaMemcpy(lig_z_d, lig_z, memsize_lig, cudaMemcpyHostToDevice);
    cudaMemcpy(ql_d, ql,memsize_ql , cudaMemcpyHostToDevice);
    cudaMemcpy(energy_d, energy, memsize_energy, cudaMemcpyHostToDevice);
      cudaMemcpy(rectype_d, rectype, memsize_recty, cudaMemcpyHostToDevice);
      cudaMemcpy(ligtype_d, ligtype, memsize_ligty, cudaMemcpyHostToDevice);
    
    
    cudaStatus = cudaMemcpyToSymbol(a_params_c,a_params,MAXTYPES * sizeof(struct autodock_param_t));
   	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
 if(cudaStatus != cudaSuccess) fprintf(stderr, "he petao %d\n", cudaStatus);
 
//Definimos hilos y bloques
    dim3 block(nconformations);
    dim3 thread(atoms_l);
    	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel 19 %d\n", cudaStatus);
	//llamamos a kernel
 
	solvation <<< block,thread>>> (atoms_r, atoms_l, nlig, rec_x_d, rec_y_d, rec_z_d, lig_x_d, lig_y_d, lig_z_d, ql_d, qr_d, energy_d,rectype_d,ligtype_d, nconformations);
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel 20 %d\n", cudaStatus);

		//control de errores kernel
	cudaDeviceSynchronize();
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Error en el kernel 12 %d\n", cudaStatus);

	//Traemos info al host
    cudaStatus = cudaMemcpy(energy, energy_d, memsize_energy, cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) fprintf(stderr, "Fallo al transferir los datos 13 %d\n", cudaStatus);

	// para comprobar que la ultima conformacion tiene el mismo resultado que la primera
	printf("Termino electrostatico de conformacion %d es: %f\n", nconformations-1, energy[nconformations-1]);

	//resultado varia repecto a SECUENCIAL y CUDA en 0.000002 por falta de precision con float
	//posible solucion utilizar double, probablemente bajara el rendimiento -> mas tiempo para calculo
	printf("Termino electrostatico %f\n", energy[0]);

	//Liberamos memoria reservada para GPU
	///cudaFree(memsize_rec);
	///cudaFree(memsize_qr);
	///cudaFree(memsize_lig);
	///cudaFree(memsize_ql);
	///cudaFree(memsize_energy);





}



/**
* Funcion que implementa la solvatacion en CPU 
*/
void forces_CPU_AU (int atoms_r, int atoms_l, int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql ,float *qr, float *energy, struct autodock_param_t *a_params, int nconformations){

	float dist, temp_desolv = 0,miatomo[3], e_desolv;
	int j,i;
	int ind1, ind2;
	int total;

	float difx,dify,difz, solv_asp_1, solv_asp_2, solv_vol_1, solv_vol_2,solv_qasp_1,solv_qasp_2;
	float  mod2x, mod2y, mod2z;	

	total = nconformations * nlig;

	for (int k=0; k < (nconformations*nlig); k+=nlig)
	{
		for(int i=0;i<atoms_l;i++){					
			e_desolv = 0;
			ind1 = ligtype[i];
			miatomo[0] = *(lig_x + k + i);
			miatomo[1] = *(lig_y + k + i);
			miatomo[2] = *(lig_z + k + i);
			solv_asp_1 = a_params[ind1].asp;
			solv_vol_1 = a_params[ind1].vol;
			for(int j=0;j<atoms_r;j++){				
				e_desolv = 0;
				ind2 = rectype[j];
				solv_asp_2 = a_params[ind2].asp;
				solv_vol_2 = a_params[ind2].vol;
				difx= (rec_x[j]) - miatomo[0];
				dify= (rec_y[j]) - miatomo[1];
				difz= (rec_z[j]) - miatomo[2];
				mod2x=difx*difx;
				mod2y=dify*dify;
				mod2z=difz*difz;
			
				difx=mod2x+mod2y+mod2z;
				dist = sqrtf(difx);
							
				e_desolv = ((solv_asp_1 * solv_vol_2) + (QASP * fabs(ql[i]) *  solv_vol_2) + (solv_asp_2 * solv_vol_1) + (QASP * fabs(qr[j]) * solv_vol_1)) * exp(-difx/(2*G_D_2));
				temp_desolv += e_desolv;	
			
			}
		}
		energy[k/nlig] = temp_desolv;
		temp_desolv = 0;
	}
	printf("Desolvation term value: %f\n",energy[0]);
}






extern void solver_AU(int mode, int atoms_r, int atoms_l,  int nlig, float *rec_x, float *rec_y, float *rec_z, float *lig_x, float *lig_y, float *lig_z, int* rectype, int* ligtype, float *ql, float *qr, float *energy_desolv, struct autodock_param_t *a_params, int nconformaciones) 
{
	double elapsed_i, elapsed_o;
	
	switch (mode) {
		case 0://Sequential execution
			printf("\* DESOLVATION TERM FUNCTION CPU MODE *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\tMode: %d\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_CPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,a_params,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("CPU Processing time: %f (seg)\n", elapsed_o);
			break;
		case 1: //OpenMP execution
			printf("\* DESOLVATION TERM FUNCTION OPENMP MODE *\n");
			printf("**************************************\n");			
			printf("Conformations: %d\tMode: %d\n",nconformaciones,mode);			
			elapsed_i = wtime();
			forces_OMP_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,a_params,nconformaciones);
			elapsed_o = wtime() - elapsed_i;
			printf ("OpenMP Processing time: %f (seg)\n", elapsed_o);
			break;
		case 2: //CUDA exeuction
            printf("\* DESOLVATION TERM FUNCTION CUDA MODE *\n");
            printf("**************************************\n");
            printf("Conformations: %d\tMode: %d\n",nconformaciones,mode);
			elapsed_i = wtime();
			forces_GPU_AU (atoms_r,atoms_l,nlig,rec_x,rec_y,rec_z,lig_x,lig_y,lig_z,rectype,ligtype,ql,qr,energy_desolv,nconformaciones,a_params);
			elapsed_o = wtime() - elapsed_i;
			printf ("GPU Processing time: %f (seg)\n", elapsed_o);			
			break; 	
	  	default:
 	     	printf("Wrong mode type: %d.  Use -h for help.\n", mode);
			exit (-1);	
	} 		
}


