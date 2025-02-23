//////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2017, Image Engine Design Inc. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//
//      * Redistributions of source code must retain the above
//        copyright notice, this list of conditions and the following
//        disclaimer.
//
//      * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided with
//        the distribution.
//
//      * Neither the name of John Haddon nor the names of
//        any other contributors to this software may be used to endorse or
//        promote products derived from this software without specific prior
//        written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//////////////////////////////////////////////////////////////////////////

#include "GafferScene/Private/IECoreScenePreview/Renderer.h"

#include "GafferScene/Private/IECoreGLPreview/AttributeVisualiser.h"
#include "GafferScene/Private/IECoreGLPreview/LightVisualiser.h"
#include "GafferScene/Private/IECoreGLPreview/LightFilterVisualiser.h"
#include "GafferScene/Private/IECoreGLPreview/ObjectVisualiser.h"
#include "GafferScene/Private/IECoreScenePreview/Placeholder.h"
#include "GafferScene/ScenePlug.h"

#include "IECoreGL/CachedConverter.h"
#include "IECoreGL/Camera.h"
#include "IECoreGL/ColorTexture.h"
#include "IECoreGL/CurvesPrimitive.h"
#include "IECoreGL/DepthTexture.h"
#include "IECoreGL/Exception.h"
#include "IECoreGL/FrameBuffer.h"
#include "IECoreGL/GL.h"
#include "IECoreGL/Group.h"
#include "IECoreGL/PointsPrimitive.h"
#include "IECoreGL/Primitive.h"
#include "IECoreGL/Renderable.h"
#include "IECoreGL/Selector.h"
#include "IECoreGL/ShaderStateComponent.h"
#include "IECoreGL/State.h"
#include "IECoreGL/ToGLCameraConverter.h"
#include "IECoreGL/IECoreGL.h"

#include "IECore/CompoundParameter.h"
#include "IECore/MessageHandler.h"
#include "IECore/PathMatcherData.h"
#include "IECore/SimpleTypedData.h"
#include "IECore/StringAlgo.h"
#include "IECore/Writer.h"

#include "Imath/ImathBoxAlgo.h"
#include "Imath/ImathMatrixAlgo.h"

#include "boost/algorithm/string/predicate.hpp"

#include "tbb/concurrent_queue.h"

#include "fmt/format.h"

#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/cameraUtil/conformWindow.h>
#include <pxr/imaging/cameraUtil/framing.h>
#include <pxr/imaging/glf/drawTarget.h>
#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/imaging/hdx/tokens.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgi/tokens.h>
#include <pxr/imaging/hgiInterop/hgiInterop.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usdImaging/usdImaging/delegate.h>
#include <pxr/usdImaging/usdImaging/sceneIndices.h>
#include <pxr/usdImaging/usdImaging/stageSceneIndex.h>
#include "pxr/imaging/glf/contextCaps.h"
#include "pxr/imaging/glf/diagnostic.h"
#include "pxr/imaging/glf/drawTarget.h"
#include "pxr/imaging/glf/glContext.h"
#include "pxr/imaging/garch/glDebugWindow.h"
#include "IECoreUSD/DataAlgo.h"
#include "IECoreScene/MeshPrimitive.h"



#include <fstream>
#include <streambuf>

#include <functional>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace std;
using namespace Imath;
using namespace IECore;
using namespace IECoreScene;
using namespace IECoreGL;
using namespace IECoreGLPreview;


#include <pxr/imaging/hd/primvarSchema.h>
#include <pxr/imaging/hd/primvarsSchema.h>
#include <pxr/imaging/hd/purposeSchema.h>
#include "pxr/imaging/hd/meshSchema.h"
#include "pxr/imaging/hd/meshTopologySchema.h"
#include <pxr/imaging/hd/retainedDataSource.h>
#include <pxr/imaging/hd/overlayContainerDataSource.h>
#include <pxr/imaging/hd/sceneIndex.h>
#include <pxr/imaging/hd/tokens.h>
#include <pxr/imaging/hd/visibilitySchema.h>
#include <pxr/imaging/hd/xformSchema.h>

#include "pxr/pxr.h"

PXR_NAMESPACE_OPEN_SCOPE

class HydraSceneIndex;

TF_DECLARE_REF_PTRS(HydraSceneIndex);

class HydraSceneIndex : public pxr::HdSceneIndexBase {
    public:
        /**
         * @brief Create a ref pointer to a grid scene index
         *
         * @return HydraSceneIndexRefPtr the ref pointer to a grid scene index
         */
        static HydraSceneIndexRefPtr New()
        {
            return pxr::TfCreateRefPtr(new HydraSceneIndex());
        }

        /**
         * @brief Construct a new grid scene index object
         *
         */
        HydraSceneIndex(){
		}


		void AddPrim(pxr::SdfPath primPath, pxr::HdSceneIndexPrim prim)
		{
			prims[primPath] = prim;
			_SendPrimsAdded({{primPath, pxr::HdPrimTypeTokens->mesh}});
		}

		void RemovePrim(pxr::SdfPath primPath)
		{
			auto it = prims.find (primPath);
			if( it != prims.end() )
    			prims.erase(it);
			
			_SendPrimsRemoved({primPath});
		}

		void SetDirtiedPrimEntries(HdSceneIndexObserver::DirtiedPrimEntries &entries)
		{
			_SendPrimsDirtied(entries);
		}

		pxr::HdSceneIndexPrim& GetPrim(const pxr::SdfPath& primPath)
		{
			return prims[primPath];
		}

        /**
         * @brief Get the prim at the given path
         *
         * @param primPath the path to a prim
         * @return pxr::HdSceneIndexPrim the hydra prim
         */
        virtual pxr::HdSceneIndexPrim GetPrim(const pxr::SdfPath& primPath) const
		{
			std::map<SdfPath, pxr::HdSceneIndexPrim>::const_iterator it;
			it = prims.find( primPath );
			if( it != prims.end() )
			{
				return it->second;
			}
    		
			return {pxr::TfToken(), nullptr};
		}

        /**
         * @brief Get the child prim paths of a prim at the specified path
         * 
         * @param primPath the path of the prim the get the child paths from
         * @return pxr::SdfPathVector a list with all child prim paths
         */
        virtual pxr::SdfPathVector GetChildPrimPaths(
            const pxr::SdfPath& primPath) const
		{
			pxr::SdfPathVector paths;

			for ( const auto &prim : prims ) {
				pxr::SdfPath parent = prim.first.GetParentPath();
				bool keepLookingParent = true;

				// if parent does not start with primPath, we skip prim
				if(strncmp(parent.GetText(), primPath.GetText(), primPath.GetString().size()))
					keepLookingParent = false;

				while(keepLookingParent)
				{
					std::cout << "checking " << prim.first << std::endl;
					std::cout << "for " << primPath << std::endl;
					std::cout << "parent: " << parent << std::endl;
					if(parent == primPath){
						// Parent of current prim == primPath.
						// We can return the prim path
						paths.push_back(prim.first);
						keepLookingParent = false;
					}
					else if( prims.find(parent) != prims.end() )
					{
						// Parent of the current prim != primPath.
						// Also, the parent of the current prim exists
						// in the map. The current prim will be returned
						// in a future call.
						keepLookingParent = false;
					}
					else{
						// Lets try with the parent of parent. This will
						// be useful in case of intermediate prim not
						// existing in the map.
						parent = parent.GetParentPath();
					}
				}
			}
			return paths;
		}

    private:
        pxr::SdfPath _primPath;
        pxr::HdSceneIndexPrim _prim;

		std::map<SdfPath, pxr::HdSceneIndexPrim> prims;
};

PXR_NAMESPACE_CLOSE_SCOPE

namespace
{
template<typename T>
T *reportedCast( const IECore::RunTimeTyped *v, const char *type, const IECore::InternedString &name )
{
	T *t = IECore::runTimeCast<T>( v );
	if( t )
	{
		return t;
	}

	IECore::msg( IECore::Msg::Warning, "IECoreGL::Renderer", fmt::format( "Expected {} but got {} for {} \"{}\".", T::staticTypeName(), v->typeName(), type, name.string() ) );
	return nullptr;
}

template<typename T>
T option( const IECore::Object *v, const IECore::InternedString &name, const T &defaultValue )
{
	if( !v )
	{
		return defaultValue;
	}
	if( auto d = reportedCast<const IECore::TypedData<T>>( v, "option", name ) )
	{
		return d->readable();
	}
	return defaultValue;
}
}

//////////////////////////////////////////////////////////////////////////
// OpenGLAttributes
//////////////////////////////////////////////////////////////////////////

namespace
{

class OpenGLAttributes : public IECoreScenePreview::Renderer::AttributesInterface
{

	public :

		OpenGLAttributes( const IECore::CompoundObject *attributes )
			:	m_frustumMode( FrustumMode::WhenSelected ), m_visualisationStateColorSpace( Visualisation::ColorSpace::Display )
		{
			const FloatData *visualiserScaleData = attributes->member<FloatData>( "gl:visualiser:scale" );
			m_visualiserScale = visualiserScaleData ? visualiserScaleData->readable() : 1.0;

			if( const StringData *drawFrustumData = attributes->member<StringData>( "gl:visualiser:frustum" ) )
			{
				if( drawFrustumData->readable() == "off" )
				{
					m_frustumMode = FrustumMode::Off;
				}
				else if( drawFrustumData->readable() == "on" )
				{
					m_frustumMode = FrustumMode::On;
				}
			}

			m_state = static_pointer_cast<const State>(
				CachedConverter::defaultCachedConverter()->convert( attributes )
			);

			IECoreGL::ConstStatePtr visualisationState;
			m_visualisations = AttributeVisualiser::allVisualisations( attributes, visualisationState );

			IECoreGL::ConstStatePtr lightVisualisationState;
			m_lightVisualisations = LightVisualiser::allVisualisations( attributes, lightVisualisationState );

			IECoreGL::ConstStatePtr lightFilterVisualisationState;
			m_lightFilterVisualisations = LightFilterVisualiser::allVisualisations( attributes, lightFilterVisualisationState );

			if( !m_lightFilterVisualisations.empty() )
			{
				if( !m_lightVisualisations.empty() )
				{
					// Light filter visualisers are in `m_lightFilterVisualisations` and light visualisers are in
					// `m_lightVisualisations`. Combine them both into `m_lightVisualisations` so that
					// filters attached to light locations are drawn as expected.
					m_lightVisualisations.insert( m_lightVisualisations.end(),
						m_lightFilterVisualisations.begin(), m_lightFilterVisualisations.end()
					);
				}
				else
				{
					// If we don't have a light visualisation, but do have filters, make sure they're drawn.
					m_lightVisualisations = m_lightFilterVisualisations;
				}
			}

			if( visualisationState || lightVisualisationState || lightFilterVisualisationState )
			{
				StatePtr combinedState = new State( /* complete = */ false );

				if( visualisationState )
				{
					combinedState->add( const_cast<State *>( visualisationState.get() ) );
				}

				if( lightVisualisationState )
				{
					combinedState->add( const_cast<State *>( lightVisualisationState.get() ) );
				}

				if( lightFilterVisualisationState )
				{
					combinedState->add( const_cast<State *>( lightFilterVisualisationState.get() ) );
				}

				m_visualisationState = combinedState;
				auto solidState = m_visualisationState->get<IECoreGL::Primitive::DrawSolid>();
				// The Visualiser API doesn't currently allow a colour space to
				// be associated with the visualisation state. So we use a
				// heuristic : if the state includes solid drawing then we
				// assume Scene space. This allows custom mesh light texture
				// visualisers to be shown with an appropriate colour transform.
				// Otherwise we assume Display space, which gives us what we
				// want for the coloured outline from our own mesh light
				// visualiser.
				if( !solidState || solidState->value() )
				{
					m_visualisationStateColorSpace = Visualisation::ColorSpace::Scene;
				}
			}
		}

		const State *state() const
		{
			return m_state.get();
		}

		const State *visualisationState( Visualisation::ColorSpace colorSpace ) const
		{
			return colorSpace == m_visualisationStateColorSpace ? m_visualisationState.get() : nullptr;
		}

		const IECoreGLPreview::Visualisations &visualisations() const
		{
			return m_visualisations;
		}

		const IECoreGLPreview::Visualisations &lightVisualisations() const
		{
			return m_lightVisualisations;
		}

		const IECoreGLPreview::Visualisations &lightFilterVisualisations() const
		{
			return m_lightFilterVisualisations;
		}

		float visualiserScale() const
		{
			return m_visualiserScale;
		}

		bool drawFrustum( bool isSelected ) const
		{
			switch ( m_frustumMode )
			{
				case FrustumMode::WhenSelected :
					return isSelected;
				case FrustumMode::On :
					return true;
				default :
					return false;
			}
		}

	private :

		ConstStatePtr m_state;
		ConstStatePtr m_visualisationState;
		Visualisations m_visualisations;
		Visualisations m_lightVisualisations;
		Visualisations m_lightFilterVisualisations;

		enum class FrustumMode : char
		{
			Off,
			WhenSelected,
			On
		};
		FrustumMode m_frustumMode;

		float m_visualiserScale = 1.0f;
		Visualisation::ColorSpace m_visualisationStateColorSpace;
};

IE_CORE_DECLAREPTR( OpenGLAttributes )

} // namespace

//////////////////////////////////////////////////////////////////////////
// HydraObject
//////////////////////////////////////////////////////////////////////////

namespace
{

class HydraObject : public IECoreScenePreview::Renderer::ObjectInterface
{

	public :

		HydraObject( const std::string &name, const IECore::Object *object, const ConstOpenGLAttributesPtr &attributes, pxr::HydraSceneIndexRefPtr sceneIndex)
			:	m_primPath(name),
				m_objectType( object ? object->typeId() : IECore::NullObjectTypeId ),
				m_attributes( attributes ),
				m_sceneIndex( sceneIndex)
		{
			m_prim = { pxr::TfToken(), nullptr };
		}

		~HydraObject() override
		{
			m_sceneIndex->RemovePrim(m_primPath);
		}

		const std::string& name()
		{
			return m_primPath.GetString();
		}

		void transform( const Imath::M44f &transform ) override
		{
			std::cout << "object transform: " << std::endl;
			auto gfTransform = pxr::GfMatrix4d( IECoreUSD::DataAlgo::toUSD( transform ) );
			SetPrimXform(gfTransform);
		}

		void transform( const std::vector<Imath::M44f> &samples, const std::vector<float> &times ) override
		{
			std::cout << "object transform sample: " << std::endl;
			transform( samples.front() );
		}

		bool attributes( const IECoreScenePreview::Renderer::AttributesInterface *attributes ) override
		{
			std::cout << "objec attribute: " << attributes << std::endl;
			ConstOpenGLAttributesPtr openGLAttributes = static_cast<const OpenGLAttributes *>( attributes );
			return true;
		}

		void link( const IECore::InternedString &type, const IECoreScenePreview::Renderer::ConstObjectSetPtr &objects ) override
		{
			std::cout << "object link: " << type << std::endl;
		}

		void assignID( uint32_t id ) override
		{
			std::cout << "object assignID: " << id << std::endl;
			// The GL renderer provides a more lightweight ID mechanism where
			// IDs are just the index in the object list, and don't need
			// assigning. This is exposed via the `gl:querySelection` command.
			// So we don't implement `assignID()` for now.
			/// \todo Evaluate overhead of the more general ID mechanism, and
			/// consider dropping the custom OpenGL one.
		}

	protected:
		pxr::SdfPath m_primPath;
		IECore::TypeId m_objectType;
		ConstOpenGLAttributesPtr m_attributes;
		pxr::HydraSceneIndexRefPtr m_sceneIndex;
		pxr::HdSceneIndexPrim m_prim;

		void SetPrimXform(pxr::GfMatrix4d xform)
		{
			auto hdXform = pxr::HdRetainedTypedSampledDataSource<pxr::GfMatrix4d>::New(
				xform
			);

			auto hdResetTransformStack = pxr::HdRetainedTypedSampledDataSource<bool>::New(
				false
			);

			pxr::HdSceneIndexPrim& prim = m_sceneIndex->GetPrim(m_primPath);

			prim.dataSource = pxr::HdOverlayContainerDataSource::New(
				pxr::HdRetainedContainerDataSource::New(
					pxr::HdXformSchemaTokens->xform,
					pxr::HdXformSchema::Builder()
							.SetMatrix(hdXform)
							.SetResetXformStack(hdResetTransformStack)
							.Build()),
				prim.dataSource);

			SetDirtiedPrimEntries(pxr::HdXformSchemaTokens->xform);
		}

		void SetDirtiedPrimEntries(pxr::TfToken token)
		{
			pxr::HdSceneIndexObserver::DirtiedPrimEntries entries;
			pxr::HdDataSourceLocator locator(token);
			entries.push_back({m_primPath, locator});

			m_sceneIndex->SetDirtiedPrimEntries(entries);
		}

		
};

IE_CORE_FORWARDDECLARE( HydraObject )

class HydraMesh : public HydraObject
{

	public :

		HydraMesh( const std::string &name, const IECoreScene::MeshPrimitive *mesh, const ConstOpenGLAttributesPtr &attributes, pxr::HydraSceneIndexRefPtr sceneIndex)
			:	HydraObject( name, mesh, attributes, sceneIndex )
		{
			auto &verticesPerFace = mesh->verticesPerFace()->readable();
			auto hdFvc = pxr::VtIntArray( verticesPerFace.begin(), verticesPerFace.end() );
			// std::cout << "fvc: " << hdFvc << std::endl;

			auto &vertexIds = mesh->vertexIds()->readable();
			auto hdFvi = pxr::VtIntArray( vertexIds.begin(), vertexIds.end() );
			// std::cout << "fvi: " << hdFvi << std::endl;

			const V3fVectorData *p = mesh->variableData<V3fVectorData>( "P", PrimitiveVariable::Vertex );
			const vector<Imath::V3f> &points = p->readable();
			pxr::VtVec3fArray pts;
			for( auto pt: points)
				pts.push_back(pxr::GfVec3f( IECoreUSD::DataAlgo::toUSD( pt ) ));
			// std::cout << "pts: " << pts << std::endl;

			m_prim = pxr::HdSceneIndexPrim(
				{
					pxr::HdPrimTypeTokens->mesh,
					pxr::HdRetainedContainerDataSource::New()
				}
			);

			sceneIndex->AddPrim(m_primPath, m_prim);

			SetPrimTopology( hdFvc, hdFvi);
			SetPrimPoints(pts);
		}

	private:

		void SetPrimPoints(pxr::VtArray<pxr::GfVec3f> points)
		{
			auto hdPoints = pxr::HdRetainedTypedSampledDataSource<pxr::VtArray<pxr::GfVec3f>>::New(
				points
			);
			auto hdRole = pxr::HdPrimvarSchema::BuildRoleDataSource(
				pxr::HdPrimvarSchemaTokens->point
			);
			auto hdInterpolation =  pxr::HdPrimvarSchema::BuildInterpolationDataSource(
				pxr::HdPrimvarSchemaTokens->varying
			);

			pxr::HdSceneIndexPrim& prim = m_sceneIndex->GetPrim(m_primPath);

			prim.dataSource = pxr::HdOverlayContainerDataSource::New(
				pxr::HdRetainedContainerDataSource::New(
					pxr::HdPrimvarsSchemaTokens->primvars,
					pxr::HdRetainedContainerDataSource::New(
						pxr::HdPrimvarsSchemaTokens->points,
							pxr::HdPrimvarSchema::Builder()
								.SetPrimvarValue(hdPoints)
								.SetRole(hdRole)
								.SetInterpolation(hdInterpolation)
								.Build())),
				prim.dataSource);

			SetDirtiedPrimEntries(pxr::HdPrimvarsSchemaTokens->primvars);
		}

		void SetPrimTopology(pxr::VtIntArray fvc, pxr::VtIntArray fvi)
		{
			auto hdFvc = pxr::HdRetainedTypedSampledDataSource<pxr::VtIntArray>::New(
				fvc
			);
			auto hdFvi = pxr::HdRetainedTypedSampledDataSource<pxr::VtIntArray>::New(
				fvi
			);

			auto hdOrientaion = pxr::HdMeshTopologySchema::BuildOrientationDataSource(
				pxr::HdMeshTopologySchemaTokens->rightHanded
			);

			pxr::HdSceneIndexPrim& prim = m_sceneIndex->GetPrim(m_primPath);

			prim.dataSource = pxr::HdOverlayContainerDataSource::New(
				pxr::HdRetainedContainerDataSource::New(
					pxr::HdMeshSchemaTokens->mesh,
					pxr::HdMeshSchema::Builder()
						.SetTopology(
							pxr::HdMeshTopologySchema::Builder()
								.SetFaceVertexCounts(hdFvc)
								.SetFaceVertexIndices(hdFvi)
								.SetOrientation(hdOrientaion)
								.Build())
						.Build()),
				prim.dataSource);

			SetDirtiedPrimEntries(pxr::HdMeshSchemaTokens->mesh);
		}
};

IE_CORE_FORWARDDECLARE( HydraMesh )

} // namespace

//////////////////////////////////////////////////////////////////////////
// HydraCamera
//////////////////////////////////////////////////////////////////////////

namespace
{

class HydraCamera : public IECoreScenePreview::Renderer::ObjectInterface
{

	public :

		HydraCamera(const std::string name): m_name(name)
		{
		}

		void fromGafferCamera(const IECoreScene::Camera *camera)
		{
			m_coreCamera = camera;

			m_gfCamera.SetFStop(camera->getFStop());
			m_gfCamera.SetFocusDistance(camera->getFocusDistance());
			m_gfCamera.SetFocalLength(camera->getFocalLength());

			const Imath::V2f planes = camera->getClippingPlanes();
			m_gfCamera.SetClippingRange({planes[0], planes[1]});

			const Imath::V2i &resolution = camera->getResolution();
			double aspectRatio = resolution[0] / (double)resolution[1];

			const Imath::V2f &fieldsOfView = camera->calculateFieldOfView();
			double horizontalFieldOfView = fieldsOfView[0];

			m_gfCamera.SetPerspectiveFromAspectRatioAndFieldOfView(
				aspectRatio,
				horizontalFieldOfView,
				pxr::GfCamera::FOVDirection::FOVHorizontal
			);
		}

		void name(std::string name)
		{
			m_name = name;
		}

		void link( const IECore::InternedString &type, const IECoreScenePreview::Renderer::ConstObjectSetPtr &objects ) override
		{
		}

		void transform( const Imath::M44f &transform ) override
		{
			auto gfTransform = pxr::GfMatrix4d( IECoreUSD::DataAlgo::toUSD( transform ) );
			m_gfCamera.SetTransform(gfTransform);
		}

		void transform( const std::vector<Imath::M44f> &samples, const std::vector<float> &times ) override
		{
			// TODO
		}

		bool attributes( const IECoreScenePreview::Renderer::AttributesInterface *attributes ) override
		{
			// Attributes don't affect the camera, so the edit always "succeeds".
			return true;
		}

		void assignID( uint32_t id ) override
		{
			/// \todo Implement me
		}

		pxr::GfCamera getGfCamera() const
		{
			return m_gfCamera;
		}

		pxr::GfVec2i getResolution() const
		{
			const V2i res = m_coreCamera->getResolution();
			return {res.x, res.y};
		}

	private :
		const IECoreScene::Camera *m_coreCamera;
		std::string m_name;
		pxr::GfCamera m_gfCamera;

};

IE_CORE_FORWARDDECLARE( HydraCamera )


} // namespace

//////////////////////////////////////////////////////////////////////////
// OpenGLLight
//////////////////////////////////////////////////////////////////////////

namespace
{

class OpenGLLight : public HydraObject
{

	public :

		OpenGLLight( const std::string &name, const IECore::Object *light, const ConstOpenGLAttributesPtr &attributes, pxr::HydraSceneIndexRefPtr sceneIndex)
			:	HydraObject( name, light, attributes, sceneIndex )
		{
		}
};

IE_CORE_FORWARDDECLARE( OpenGLLight )

class OpenGLLightFilter : public HydraObject
{

	public :

		OpenGLLightFilter( const std::string &name, const IECore::Object *object, const ConstOpenGLAttributesPtr &attributes, pxr::HydraSceneIndexRefPtr sceneIndex)
			:	HydraObject( name, object, attributes, sceneIndex )
		{
		}
};

IE_CORE_FORWARDDECLARE( OpenGLLightFilter )

} // namespace
//////////////////////////////////////////////////////////////////////////
// HydraRenderer
//////////////////////////////////////////////////////////////////////////

namespace
{

class HydraRenderer final : public IECoreScenePreview::Renderer
{

	public :

		HydraRenderer( RenderType renderType, const std::string &fileName, const IECore::MessageHandlerPtr &messageHandler )
			:	m_renderType( renderType ), m_baseStateOptions( new CompoundObject ),
				m_renderObjects( true ), m_messageHandler( messageHandler )
		{
			std::cout << "init HydraRenderer" << std::endl;
			if( renderType == SceneDescription )
			{
				throw IECore::Exception( "Unsupported render type" );
			}

			sceneIndex = pxr::HydraSceneIndex::New();
		}

		~HydraRenderer() override
		{
		}

		IECore::InternedString name() const override
		{
			return "Hydra";
		}

		void option( const IECore::InternedString &name, const IECore::Object *value ) override
		{
			std::cout << "option: " << name << std::endl;

			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			if( name == "camera" )
			{
				m_camera = ::option<string>( value, name, "" );
			}
			else if( name == "frame" || name == "sampleMotion" )
			{
				// We know what these mean, we just have no use for them.
			}
			else if( name == "gl:selection" )
			{
				m_selection = ::option<IECore::PathMatcher>( value, name, IECore::PathMatcher() );
			}
			else if(
				boost::starts_with( name.string(), "gl:primitive:" ) ||
				boost::starts_with( name.string(), "gl:pointsPrimitive:" ) ||
				boost::starts_with( name.string(), "gl:curvesPrimitive:" ) ||
				boost::starts_with( name.string(), "gl:smoothing:" )
			)
			{
				if( value )
				{
					m_baseStateOptions->members()[name] = value->copy();
				}
				else
				{
					m_baseStateOptions->members().erase( name );
				}
				m_baseState = nullptr; // We'll update it lazily in `baseState()`
			}
			/// \todo We can't support this being modified after the scene has
			/// already been generated, because we've thrown away the source
			/// objects already. This is similar to `ai:ignore_subdivision`.
			/// Perhaps `option()` should have a return value to indicate to the
			/// RenderController that it needs to resend objects?
			else if( name == "gl:renderObjects" )
			{
				m_renderObjects = ::option<bool>( value, name, true );
			}
			else if( boost::contains( name.string(), ":" ) && !boost::starts_with( name.string(), "gl:" ) )
			{
				// Ignore options prefixed for some other renderer.
			}
			else
			{
				IECore::msg( IECore::Msg::Warning, "IECoreGL::Renderer::option", fmt::format( "Unknown option \"{}\".", name.string() ) );
			}
		}

		void output( const IECore::InternedString &name, const Output *output ) override
		{
			std::cout << "output: " << name << std::endl;

			if( output )
			{
				m_outputs[name] = output;
			}
			else
			{
				m_outputs.erase( name );
			}
		}

		Renderer::AttributesInterfacePtr attributes( const IECore::CompoundObject *attributes ) override
		{
			std::cout << "attribute: " << attributes->baseTypeName() << std::endl;

			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			OpenGLAttributesPtr result = new OpenGLAttributes( attributes );
			return result;
		}

		ObjectInterfacePtr camera( const std::string &name, const IECoreScene::Camera *camera, const AttributesInterface *attributes ) override
		{
			std::cout << "camera: " << name << std::endl;
			std::cout << "camera attr: " << attributes << std::endl;

			HydraCameraPtr hdCam;
			std::unordered_map<string, HydraCameraPtr>::const_iterator it;
			it = m_cameras.find( name );
			if( it != m_cameras.end() )
			{
				hdCam = it->second;
			}
			else
			{
				hdCam = new HydraCamera(name);
				m_cameras[name] = hdCam;
			}

			hdCam->fromGafferCamera(camera);

			return hdCam;
		}

		ObjectInterfacePtr light( const std::string &name, const IECore::Object *object, const AttributesInterface *attributes ) override
		{
			std::cout << "light: " << name << std::endl;
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			OpenGLLightPtr result = new OpenGLLight( name, object, static_cast<const OpenGLAttributes *>( attributes ), sceneIndex );
			return result;
		}

		ObjectInterfacePtr lightFilter( const std::string &name, const IECore::Object *object, const AttributesInterface *attributes ) override
		{
			std::cout << "lightfilter: " << name << std::endl;
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			OpenGLLightFilterPtr result = new OpenGLLightFilter( name, object, static_cast<const OpenGLAttributes *>( attributes ), sceneIndex );
			return result;
		}

		Renderer::ObjectInterfacePtr object( const std::string &name, const IECore::Object *object, const AttributesInterface *attributes ) override
		{
			std::cout << "object: " << name << std::endl;
			std::cout << "object attr: " << attributes->refCount() << std::endl;

			HydraObjectPtr hdObj;
			std::unordered_map<string, HydraObjectPtr>::const_iterator it;
			it = m_objects.find( name );
			if( it != m_objects.end() )
			{
				std::cout << "object found" << std::endl;
				hdObj = it->second;
			}
			else if (object->typeId() == MeshPrimitive::staticTypeId())
			{
				std::cout << "create Mesh" << std::endl;
				auto *mesh = dynamic_cast<const IECoreScene::MeshPrimitive*>(object);
				hdObj = new HydraMesh(name, mesh, static_cast<const OpenGLAttributes *>( attributes ), sceneIndex );
				m_objects[name] = hdObj;
			}
			else
			{
				std::cout << "unhandled object of type: " << object->typeId() <<std::endl;
				hdObj = new HydraObject(name, object, static_cast<const OpenGLAttributes *>( attributes ), sceneIndex );
				m_objects[name] = hdObj;
			}

			return hdObj;
		}

		ObjectInterfacePtr object( const std::string &name, const std::vector<const IECore::Object *> &samples, const std::vector<float> &times, const AttributesInterface *attributes ) override
		{
			std::cout << "object1: " << name << std::endl;

			return object( name, samples.front(), attributes );
		}

		void render() override
		{
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			if( m_renderType == Interactive )
			{
				// We currently don't have any use for interactively rendering
				// to outputs defined by the `output()` function. To facilitate
				// interactive use in a CompoundRenderer (where the other
				// renderer _is_ rendering to the outputs), we instead define a
				// separate `gl:renderToCurrentContext` command which renders
				// into a framebuffer managed by the client.
			}
			else
			{
				renderBatch();
			}
		}

		void pause() override
		{
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			if( m_renderType != Interactive )
			{
				IECore::msg( IECore::Msg::Warning, "IECoreGL::Renderer::pause", "Cannot pause non-interactive renders" );
			}
		}

		IECore::DataPtr command( const IECore::InternedString name, const IECore::CompoundDataMap &parameters ) override
		{
			std::cout << "command: " << name << std::endl;

			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			if( name == "gl:queryBound" )
			{
				return queryBound( parameters );
			}
			else if( name == "gl:querySelection" )
			{
				return querySelectedObjects( parameters );
			}
			else if( name == "gl:renderToCurrentContext" )
			{
				renderToCurrentContext( parameters );
				return nullptr;
			}
			else if( boost::starts_with( name.string(), "gl:" ) || name.string().find( ":" ) == string::npos )
			{
				IECore::msg( IECore::Msg::Warning, "IECoreGL::Renderer::command", fmt::format( "Unknown command \"{}\".", name.string() ) );
			}

			return nullptr;
		}

	private :

		void renderToCurrentContext( const IECore::CompoundDataMap &parameters )
		{
			removeDeletedObjects();

			GLint prevProgram;
			glGetIntegerv( GL_CURRENT_PROGRAM, &prevProgram );
			glPushAttrib( GL_ALL_ATTRIB_BITS );

			if(!hgi){
				std::cout << "init hgi" << std::endl;

				hgi = pxr::Hgi::CreatePlatformDefaultHgi();
				hgiDriver.name = pxr::HgiTokens->renderDriver;
				hgiDriver.driver = pxr::VtValue(hgi.get());
		
				// init draw target

				pxr::HdRendererPluginRegistry& registry =
					pxr::HdRendererPluginRegistry::GetInstance();

				pxr::TfToken resolvedId = registry.GetDefaultPluginId(true);
				renderDelegate = registry.CreateRenderDelegate(resolvedId);

				// Use the render delegate ptr (rather than 'this' ptr) for generating
				// the unique id.
				const std::string renderInstanceId =
					pxr::TfStringPrintf("UsdImagingGLEngineX_%s_%p", resolvedId.GetText(),
										(void*)renderDelegate.Get());

				// Recreate the render index
				renderIndex = pxr::HdRenderIndex::New(renderDelegate.Get(), {&hgiDriver},
													renderInstanceId);


				renderIndex->InsertSceneIndex(sceneIndex,
											pxr::SdfPath("/stageSceneIndex"));

				// set task controller
				taskController = new pxr::HdxTaskController(
					renderIndex, pxr::SdfPath("/stageSceneIndex"), true);

				taskController->SetRenderOutputs({pxr::HdAovTokens->color});

				taskController->SetOverrideWindowPolicy(
					pxr::CameraUtilConformWindowPolicy::CameraUtilMatchVertically);

				auto cam_pos = pxr::GfVec3f(10, 10, 10);
				auto sceneAmbient = pxr::GfVec4f(0.01, 0.01, 0.01, 1.0);
				auto material = pxr::GlfSimpleMaterial();
				auto lights = pxr::GlfSimpleLightVector();

				auto l = pxr::GlfSimpleLight();
				l.SetAmbient(pxr::GfVec4f(0, 0, 0, 0));
				l.SetPosition(pxr::GfVec4f(cam_pos[0], cam_pos[1], cam_pos[2], 1));
				lights.push_back(l);

				material.SetAmbient(pxr::GfVec4f(0.2, 0.2, 0.2, 1.0));
				material.SetSpecular(pxr::GfVec4f(0.1, 0.1, 0.1, 1.0));
				material.SetShininess(32.0);

				pxr::GlfSimpleLightingContextRefPtr lightingContextForOpenGLState =
					pxr::GlfSimpleLightingContext::New();

				lightingContextForOpenGLState->SetLights(lights);
				lightingContextForOpenGLState->SetMaterial(material);
				lightingContextForOpenGLState->SetSceneAmbient(sceneAmbient);
				lightingContextForOpenGLState->SetUseLighting(lights.size() > 0);

				taskController->SetLightingState(lightingContextForOpenGLState);

				// render

				// XXX(UsdImagingPaths): This bit is weird: we get the stage from "root",
				// gate population by _rootPath (which may be different), and then pass
				// root.GetPath() to hydra as the root to draw from. Note that this
				// produces incorrect results in UsdImagingDelegate for native instancing.
				const pxr::SdfPathVector paths = {pxr::SdfPath("/stageSceneIndex")};

				// init collection
				pxr::HdRprimCollection renderCollection;
				renderCollection = pxr::HdRprimCollection(
					pxr::HdTokens->geometry,
					pxr::HdReprSelector(pxr::HdReprTokens->smoothHull));
				renderCollection.SetRootPaths(paths);
				taskController->SetCollection(renderCollection);

				pxr::TfTokenVector renderTags;
				renderTags.clear();
				renderTags.reserve(4);
				renderTags.push_back(pxr::HdRenderTagTokens->geometry);
				renderTags.push_back(pxr::HdRenderTagTokens->proxy);
				taskController->SetRenderTags(renderTags);

				pxr::HdxRenderTaskParams params;
				params.enableLighting = true;
				taskController->SetRenderParams(params);

				// clear color to the color AOV.
				pxr::GfVec4f clearColor = pxr::GfVec4f(1.f, .1f, .1f, 1.0f);
				pxr::HdAovDescriptor colorAovDesc =
					taskController->GetRenderOutputSettings(pxr::HdAovTokens->color);
				if (colorAovDesc.format != pxr::HdFormatInvalid) {
					colorAovDesc.clearValue = pxr::VtValue(clearColor);
					taskController->SetRenderOutputSettings(pxr::HdAovTokens->color,
															colorAovDesc);
				}

				taskController->SetEnableSelection(true);

				pxr::HdxSelectionTrackerSharedPtr selTracker =
					std::make_shared<pxr::HdxSelectionTracker>();

				pxr::VtValue selectionValue(selTracker);
				engine.SetTaskContextData(pxr::HdxTokens->selectionState, selectionValue);
			}

			HydraCameraPtr camera;
			if( m_camera != "" )
			{
				std::unordered_map<string, HydraCameraPtr>::const_iterator it;
				it = m_cameras.find( m_camera );
				if( it != m_cameras.end() )
				{
					camera = it->second;
				}
			}

			const pxr::GfVec2i resolution = camera->getResolution();
			int width = resolution[0] * 2;
			int height = resolution[1] * 2;


			taskController->SetRenderViewport(pxr::GfVec4f(0, 0, width, height));
			taskController->SetRenderBufferSize(pxr::GfVec2i(width, height));

			pxr::GfRange2f displayWindow(pxr::GfVec2f(0, 0), pxr::GfVec2f(width, height));
			pxr::GfRect2i renderBufferRect(pxr::GfVec2i(0, 0), width, height);
			pxr::GfRect2i dataWindow = renderBufferRect.GetIntersection(renderBufferRect);
			pxr::CameraUtilFraming framing(displayWindow, dataWindow);

			taskController->SetFraming(framing);

			// Imath::M44f transform = camera->getTransform();

			// // IECoreGL::Camera* cam = camera->camera();
			// auto r = pxr::GfMatrix4d( IECoreUSD::DataAlgo::toUSD( transform ) );

			auto gfCam = camera->getGfCamera();
			auto frustum = gfCam.GetFrustum();

			taskController->SetFreeCameraMatrices(frustum.ComputeViewMatrix(),
												frustum.ComputeProjectionMatrix());

			// render

			pxr::HdTaskSharedPtrVector tasks = taskController->GetRenderingTasks();
			engine.Execute(renderIndex, &tasks);

			glPopAttrib();
			glUseProgram( prevProgram );
		}

		void renderBatch()
		{
			
		}

		void processQueue()
		{
		}

		// During interactive renders, the client code controls the lifetime
		// of objects by managing ObjectInterfacePtrs. But we also hold a
		// reference to the objects ourselves so we can iterate to render them.
		// Here we remove any objects with only a single reference - our own.
		// This does mean we delete objects later than the client might expect,
		// but this is actually necessary anyway, because we can only delete GL
		// resources on the main thread.
		void removeDeletedObjects()
		{
			for( auto it = m_cameras.begin(); it != m_cameras.end(); )
			{
				// Cameras are referenced by both m_cameras and m_objects
				if( it->second->refCount() <= 1 )
				{
					std::cout << "erase cam" << std::endl;
					it = m_cameras.erase( it );
				}
				else
				{
					++it;
				}
			}

			for( auto it = m_objects.begin(); it != m_objects.end(); )
			{
				if( it->second->refCount() <= 1 )
				{
					std::cout << "erase obj" << std::endl;
					it = m_objects.erase( it );
				}
				else
				{
					++it;
				}
			}
		}

		void renderObjects( IECoreGL::State *currentState, Visualisation::ColorSpace colorSpace )
		{
		}

		void writeOutputs( const FrameBuffer *frameBuffer )
		{
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			for( const auto &namedOutput : m_outputs )
			{
				IECoreImage::ImagePrimitivePtr image = nullptr;
				const string &data = namedOutput.second->getData();
				if( data == "rgba" )
				{
					image = frameBuffer->getColor()->imagePrimitive();
				}
				else if( data == "rgb" )
				{
					image = frameBuffer->getColor()->imagePrimitive();
					image->channels.erase( "A" );
				}
				else if( data == "z" )
				{
					image = frameBuffer->getDepth()->imagePrimitive();
				}
				else
				{
					IECore::msg( IECore::Msg::Warning, "IECoreGL::Renderer", fmt::format( "Unsupported data format \"{}\".", data ) );
					return;
				}

				const string &type = namedOutput.second->getType();
				IECore::WriterPtr writer = IECore::Writer::create( image, "tmp." + type );
				if( !writer )
				{
					IECore::msg( IECore::Msg::Warning, "IECoreGL::Renderer", fmt::format( "Unsupported display type \"{}\".", type ) );
					return;
				}

				writer->parameters()->parameter<IECore::FileNameParameter>( "fileName" )->setTypedValue( namedOutput.second->getName() );
				writer->write();
			}
		}

		DataPtr queryBound( const CompoundDataMap &parameters )
		{
			Box3f result;
			return new Box3fData( result );
		}

		DataPtr querySelectedObjects( const CompoundDataMap &parameters )
		{
			PathMatcher result;
			return new PathMatcherData( result );
		}

		IECoreGL::State *baseState()
		{
			if( !m_baseState )
			{
				m_baseState = new IECoreGL::State( /* complete = */ true );
				IECoreGL::ConstStatePtr optionsState = static_pointer_cast<const State>(
					CachedConverter::defaultCachedConverter()->convert( m_baseStateOptions.get() )
				);
				m_baseState->add( const_pointer_cast<State>( optionsState ) );
			}
			return m_baseState.get();
		}

		// Global options
		RenderType m_renderType;
		string m_camera;
		IECore::PathMatcher m_selection;
		IECore::CompoundObjectPtr m_baseStateOptions;
		IECoreGL::StatePtr m_baseState;
		bool m_renderObjects;

		IECore::MessageHandlerPtr m_messageHandler;




		unordered_map<InternedString, ConstOutputPtr> m_outputs;
		std::unordered_map<string, HydraCameraPtr> m_cameras;
		HydraCameraPtr m_cam = nullptr;

		std::unordered_map<string, HydraObjectPtr> m_objects;

		using OpenGLAttributesVector = std::vector<OpenGLAttributesPtr>;
		OpenGLAttributesVector m_attributes;

		// Registration with factory
		static Renderer::TypeDescription<HydraRenderer> g_typeDescription;

		pxr::GlfDrawTargetRefPtr drawTarget;
		pxr::HdxTaskController* taskController;
		pxr::HdEngine engine;
		
		pxr::HdRenderIndex* renderIndex;
		pxr::HgiUniquePtr hgi;
		pxr::HdDriver hgiDriver;
		pxr::HdPluginRenderDelegateUniqueHandle renderDelegate;

		pxr::HydraSceneIndexRefPtr sceneIndex;

};

IECoreScenePreview::Renderer::TypeDescription<HydraRenderer> HydraRenderer::g_typeDescription( "Hydra" );

} // namespace
