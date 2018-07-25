#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cmath>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "NvInferPlugin.h"
#include <sys/time.h>
#include <fstream>


using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

//static const int OUTPUT_SIZE = 1;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME1 = "detection_out";
const char* OUTPUT_BLOB_NAME2 = "detection_out2";


long long ustime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec)*1000000;
    ust += tv.tv_usec;
    return ust;
}

/* Return the UNIX time in milliseconds */
long long mstime(void) {
    return ustime()/1000;
}

template<int OutC>
class SliceLayer : public IPlugin
{
public:
    SliceLayer(){}
    SliceLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return OutC; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(3 == inputs[0].nbDims);
        return DimsCHW(1, inputs[0].d[1], inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(outputs[1],inputs[0]+1*batchSize*_size*sizeof(float),batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(outputs[2],inputs[0]+2*batchSize*_size*sizeof(float),batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        cudaMemcpyAsync(outputs[3],inputs[0]+3*batchSize*_size*sizeof(float),batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};

class FlattenLayer : public IPlugin
{
public:
    FlattenLayer(){}
    FlattenLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];

        return DimsCHW(_size, 1, 1);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void* const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream);
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};

template<int OutC>
class Reshape : public IPlugin
{
public:
    Reshape() {}
    Reshape(const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override
    {
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
        return DimsCHW(inputs[0].d[0] * inputs[0].d[1] / OutC,OutC,inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream);
        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }

    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};

class PluginFactory: public nvinfer1::IPluginFactory,
                      public nvcaffeparser1::IPluginFactory {
public:
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override  ;
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override ; 
        bool isPlugin(const char* name) override;
        void destroyPlugin();
private:
    std::map<std::string, plugin::INvPlugin*> _nvPlugins; 
};



bool PluginFactory::isPlugin(const char* name)
{
	return ( !strcmp(name,"conv3_3_norm_mbox_loc_perm")
            || !strcmp(name,"conv3_3_norm_mbox_loc_flat")
            || !strcmp(name,"conv3_3_norm_mbox_conf_slice")
            || !strcmp(name,"conv3_3_norm_mbox_conf_out")
            || !strcmp(name,"conv3_3_norm_mbox_conf_perm")
            || !strcmp(name,"conv3_3_norm_mbox_conf_flat")
            || !strcmp(name,"conv3_3_norm_mbox_priorbox")
           ||!strcmp(name,"conv4_3_norm_mbox_loc_perm")
           || !strcmp(name,"conv4_3_norm_mbox_conf_perm")
           || !strcmp(name,"conv4_3_norm_mbox_priorbox")
           || !strcmp(name,"conv5_3_norm_mbox_loc_perm")
           || !strcmp(name,"conv5_3_norm_mbox_conf_perm")
           || !strcmp(name,"conv5_3_norm_mbox_priorbox")
           || !strcmp(name,"fc7_mbox_loc_perm")
           || !strcmp(name,"fc7_mbox_conf_perm")
           || !strcmp(name,"fc7_mbox_priorbox")
           || !strcmp(name,"mbox_conf_reshape")
           || !strcmp(name,"mbox_loc")
           || !strcmp(name,"mbox_conf")
           || !strcmp(name,"mbox_priorbox")
           || !strcmp(name,"conv4_3_norm_mbox_loc_flat")
           || !strcmp(name,"conv4_3_norm_mbox_conf_flat")
           || !strcmp(name,"conv5_3_norm_mbox_loc_flat")
           || !strcmp(name,"conv5_3_norm_mbox_conf_flat")
           || !strcmp(name,"fc7_mbox_loc_flat")
           || !strcmp(name,"fc7_mbox_conf_flat")
           || !strcmp(name,"mbox_conf_flatten")
           || !strcmp(name,"detection_out")
	       );
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
	assert(isPlugin(layerName));

    if(! strcmp(layerName,"conv4_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv4_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"fc7_mbox_loc_perm")
             || !strcmp(layerName,"fc7_mbox_conf_perm")
             || !strcmp(layerName,"conv3_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv3_3_norm_mbox_loc_perm")
    )
    {
        _nvPlugins[layerName] = plugin::createSSDPermutePlugin(Quadruple({0,2,3,1}));  
        return _nvPlugins.at(layerName);
    }
    else if (!strcmp(layerName,"conv3_3_norm_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {16.0f};   
         
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
         
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 8.0f;  
        params.stepW = 8.0f;  
        params.offset = 0.5f;  
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  
        return _nvPlugins.at(layerName);
    }
    else if(! strcmp(layerName,"conv4_3_norm_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {32.0f};   
           
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
          
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 8.0f;  
        params.stepW = 8.0f;  
        params.offset = 0.5f;  

        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"conv5_3_norm_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {64.0f};   
           
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
        
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 16.0f;  
        params.stepW = 16.0f;  
        params.offset = 0.5f;  
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

        return _nvPlugins.at(layerName);
    }
    else if( !strcmp(layerName,"fc7_mbox_priorbox")){
        plugin::PriorBoxParameters params = {0};  
        float minSize[1] = {128.0f};   
        //float maxSize[1] = {60.0f};   
        float aspectRatios[1] = {1.0f};   
        params.minSize = (float*)minSize;  
        //params.maxSize = (float*)maxSize;  
        params.aspectRatios = (float*)aspectRatios;  
        params.numMinSize = 1;  
        params.numMaxSize = 0;  
        params.numAspectRatios = 1;  
        params.flip = false;  
        params.clip = false;  
        params.variance[0] = 0.1;  
        params.variance[1] = 0.1;  
        params.variance[2] = 0.2;  
        params.variance[3] = 0.2;  
        params.imgH = 0;  
        params.imgW = 0;  
        params.stepH = 32.0f;  
        params.stepW = 32.0f;  
        params.offset = 0.5f;  
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(params);  

        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"detection_out")){
        plugin::DetectionOutputParameters params {true, false, 0, 2, 200, 200, 0.05, 0.3, plugin::CodeTypeSSD::CENTER_SIZE, {0, 1, 2}, false, true};
        _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(params);  
        return _nvPlugins.at(layerName); 
    }
    else if(!strcmp(layerName,"mbox_conf_reshape")){
    	assert(nbWeights == 0 && weights == nullptr);
    	_nvPlugins[layerName] = (plugin::INvPlugin*)(new Reshape<2>());
    	return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_loc")
    	     || !strcmp(layerName,"mbox_conf")
             || !strcmp(layerName,"conv3_3_norm_mbox_conf_out")){
    	_nvPlugins[layerName] = plugin::createConcatPlugin(1,false);
    	return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_priorbox")){
        _nvPlugins[layerName] = plugin::createConcatPlugin(2,false);
        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv4_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"conv3_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"conv3_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv5_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv5_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"fc7_mbox_loc_flat")
           || !strcmp(layerName,"fc7_mbox_conf_flat")
           || !strcmp(layerName,"mbox_conf_flatten")
           )
    {
    	_nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer());
    	return _nvPlugins.at(layerName);
    }
    else if (!strcmp(layerName,"conv3_3_norm_mbox_conf_slice")){
        _nvPlugins[layerName] = (plugin::INvPlugin*)(new SliceLayer<4>());
        return _nvPlugins.at(layerName);
    }
    else{  
        std::cout << "warning : " << layerName << std::endl;
        assert(0);  
        return nullptr;  
    }  
}


IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
	assert(isPlugin(layerName));
    if(! strcmp(layerName,"conv4_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv4_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_loc_perm")
             || !strcmp(layerName,"conv5_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"fc7_mbox_loc_perm")
             || !strcmp(layerName,"fc7_mbox_conf_perm")
             || !strcmp(layerName,"conv3_3_norm_mbox_conf_perm")
             || !strcmp(layerName,"conv3_3_norm_mbox_loc_perm")
    ){
        _nvPlugins[layerName] = plugin::createSSDPermutePlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
	}
    else if(! strcmp(layerName,"conv4_3_norm_mbox_priorbox")
            || !strcmp(layerName,"conv5_3_norm_mbox_priorbox")
            || !strcmp(layerName,"fc7_mbox_priorbox")
            || !strcmp(layerName,"conv3_3_norm_mbox_priorbox")
            )
    {
        _nvPlugins[layerName] = plugin::createSSDPriorBoxPlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
    }
    else if(! strcmp(layerName,"detection_out")){
        _nvPlugins[layerName] = plugin::createSSDDetectionOutputPlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_conf_reshape")){
    	_nvPlugins[layerName] = (plugin::INvPlugin*)(new Reshape<2>(serialData,serialLength));
    	return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"conv4_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv4_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"conv5_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv5_3_norm_mbox_conf_flat")
           || !strcmp(layerName,"fc7_mbox_loc_flat")
           || !strcmp(layerName,"fc7_mbox_conf_flat")
           || !strcmp(layerName,"mbox_conf_flatten")
           || !strcmp(layerName,"conv3_3_norm_mbox_loc_flat")
           || !strcmp(layerName,"conv3_3_norm_mbox_conf_flat")
           )
    {
    	_nvPlugins[layerName] = (plugin::INvPlugin*)(new FlattenLayer(serialData,serialLength));
    	return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_loc")
             || !strcmp(layerName,"mbox_conf")
             || !strcmp(layerName,"conv3_3_norm_mbox_conf_out")){
         _nvPlugins[layerName] = plugin::createConcatPlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
    }
    else if(!strcmp(layerName,"mbox_priorbox")){
         _nvPlugins[layerName] = plugin::createConcatPlugin(serialData,serialLength);
        return _nvPlugins.at(layerName);
    }
    else if (!strcmp(layerName,"conv3_3_norm_mbox_conf_slice")){
        _nvPlugins[layerName] = (plugin::INvPlugin*)(new SliceLayer<4>(serialData,serialLength));
        return _nvPlugins.at(layerName);
    }
    else{  
         assert(0);  
         return nullptr;  
    }  
}


void PluginFactory::destroyPlugin()  
{  
    for (auto it=_nvPlugins.begin(); it!=_nvPlugins.end(); ++it){  
        //std::cout<<it->first<<std::endl;
        if( !strcmp(it->first.c_str(),"conv4_3_norm_mbox_loc_perm")
            || !strcmp(it->first.c_str(),"conv4_3_norm_mbox_conf_perm")
            || !strcmp(it->first.c_str(),"conv5_3_norm_mbox_loc_perm")
            || !strcmp(it->first.c_str(),"conv5_3_norm_mbox_conf_perm")
            || !strcmp(it->first.c_str(),"fc7_mbox_loc_perm")
            || !strcmp(it->first.c_str(),"fc7_mbox_conf_perm")
            || !strcmp(it->first.c_str(),"conv4_3_norm_mbox_priorbox")
            || !strcmp(it->first.c_str(),"conv5_3_norm_mbox_priorbox")
            || !strcmp(it->first.c_str(),"fc7_mbox_priorbox")
            || !strcmp(it->first.c_str(),"detection_out")
            || !strcmp(it->first.c_str(),"mbox_loc")
            || !strcmp(it->first.c_str(),"mbox_priorbox")
            || !strcmp(it->first.c_str(),"conv3_3_norm_mbox_loc_perm")
            || !strcmp(it->first.c_str(),"conv3_3_norm_mbox_conf_out")
            || !strcmp(it->first.c_str(),"conv3_3_norm_mbox_conf_perm")
            || !strcmp(it->first.c_str(),"conv3_3_norm_mbox_priorbox")
            ){
            it->second->destroy();  
        }
        _nvPlugins.erase(it);  
    } 
}


void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,
					 nvcaffeparser1::IPluginFactory* pluginFactory,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);
  
	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
															  modelFile.c_str(),
															  *network,
															  DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(100 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);  
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* detection_out,float* detection_out_2, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	
    void* buffers[3];
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);
     
    DimsCHW inputDims = static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex)), 
            outputDims1 = static_cast<DimsCHW&&>(engine.getBindingDimensions(outputIndex1)),
            outputDims2 = static_cast<DimsCHW&&>(engine.getBindingDimensions(outputIndex2));


	// create GPU buffers and a stream
	cudaMalloc(&buffers[inputIndex], batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float));
	cudaMalloc(&buffers[outputIndex1], batchSize * outputDims1.c() * outputDims1.h() * outputDims1.w() * sizeof(float));
    cudaMalloc(&buffers[outputIndex2], batchSize * outputDims2.c() * outputDims2.h() * outputDims2.w() * sizeof(float));

	cudaStream_t stream;
	cudaStreamCreate(&stream);

    long long llStart = mstime();

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	cudaMemcpyAsync(buffers[inputIndex], input, batchSize * inputDims.c() * inputDims.h() * inputDims.w()  * sizeof(float), cudaMemcpyHostToDevice, stream);
	context.enqueue(batchSize, buffers, stream, nullptr);
	cudaMemcpyAsync(detection_out, buffers[outputIndex1], batchSize * outputDims1.c() * outputDims1.h() * outputDims1.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(detection_out_2, buffers[outputIndex2], batchSize * outputDims2.c() * outputDims2.h() * outputDims2.w() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
	cudaStreamSynchronize(stream);

    long long llEnd = mstime();
    std::cout<<llEnd-llStart<<"ms"<<std::endl;

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	cudaFree(buffers[inputIndex]);
	cudaFree(buffers[outputIndex1]);
    cudaFree(buffers[outputIndex2]);
}


int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	

	const char* model_def = argv[1];
    const char* weights_def = argv[2];
    const char* image_name = argv[3];
    const char* image_name1 = argv[4];
    PluginFactory pluginFactory;
    
    IHostMemory *gieModelStream{nullptr};
   	caffeToGIEModel(model_def,weights_def, std::vector < std::string > { OUTPUT_BLOB_NAME1,OUTPUT_BLOB_NAME2}, 1,&pluginFactory, gieModelStream);
    pluginFactory.destroyPlugin();
	
    std::cout << "RT init done!" << std::endl;
    
    std::ofstream out("engine.binary",std::ios::out|std::ios::binary);
    out.write((const char*)(gieModelStream->data()),gieModelStream->size());
    out.close();
    

    int engine_buffer_size;

    std::ifstream in("engine.binary",std::ios::in | std::ios::binary);
    in.seekg(0,std::ios::end);
    engine_buffer_size = in.tellg();
    in.seekg(0,std::ios::beg);
    std::shared_ptr<char> engine_buffer {new char[engine_buffer_size]};
    in.read(engine_buffer.get(),engine_buffer_size);
    in.close();

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(engine_buffer.get(), engine_buffer_size, &pluginFactory);
    std::cout << "RT deserialize done!" << std::endl;

    if (gieModelStream) 
    {
        gieModelStream->destroy();
        gieModelStream = nullptr;
    }

	IExecutionContext *context = engine->createExecutionContext();
    std::cout << "RT createExecutionContext done!" << std::endl;

    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME), 
        outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1),
        outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2);
     
    DimsCHW inputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(inputIndex)), 
            outputDims1 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex1)),
            outputDims2 = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex2));
    
	// run inference
	float *detection_out = new float[outputDims1.c() * outputDims1.h() *outputDims1.w()];
    float *detection_out_2 = new float[outputDims2.c() * outputDims2.h() *outputDims2.w()];  

    cv::Mat img = cv::imread(image_name,1);
    cv::Size dsize = cv::Size(inputDims.h(),inputDims.w());
    cv::Mat imgResize;
    cv::resize(img, imgResize, dsize, 0, 0 , cv::INTER_LINEAR);

    cv::Mat img1 = cv::imread(image_name1,1);
    cv::Mat imgResize1;
    cv::resize(img1, imgResize1, dsize, 0, 0 , cv::INTER_LINEAR);

    float means[3] = {104.0, 117.0, 123.0};

    float *data = new float[inputDims.c() * inputDims.h() * inputDims.w()];

    for (int i = 0; i < imgResize.rows; ++i){
        for (int j = 0; j < imgResize.cols; ++j){
            data[0*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[0]) - means[0];
            data[1*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[1]) - means[1];
            data[2*inputDims.h() * inputDims.w() + i * inputDims.w() + j] = static_cast<float>(imgResize.at<cv::Vec3b>(i,j)[2]) - means[2];
        }
    }

    
	doInference(*context, data, detection_out, detection_out_2, 1);

    delete[] data;
    delete[] detection_out;
    delete[] detection_out_2;

    context->destroy();
    engine->destroy();
    runtime->destroy();
    pluginFactory.destroyPlugin();

    return 0;
}
