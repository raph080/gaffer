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

class CubeSceneIndex;

TF_DECLARE_REF_PTRS(CubeSceneIndex);

class CubeSceneIndex : public pxr::HdSceneIndexBase {
    public:
        /**
         * @brief Create a ref pointer to a grid scene index
         *
         * @return CubeSceneIndexRefPtr the ref pointer to a grid scene index
         */
        static CubeSceneIndexRefPtr New()
        {
            return pxr::TfCreateRefPtr(new CubeSceneIndex());
        }

        /**
         * @brief Construct a new grid scene index object
         *
         */
        CubeSceneIndex(){
			_primPath = pxr::SdfPath("/Cube");
			_prim = _CreateCubePrim();
			Populate(true);
		}

        /**
         * @brief Populate the grid scene index
         *
         * @param populate true to populate the scene index, false otherwise
         */
        void Populate(bool populate)
		{
			if (populate && !_isPopulated) {
				_SendPrimsAdded({{_primPath, pxr::HdPrimTypeTokens->mesh}});
			}
			else if (!populate && _isPopulated) {
				_SendPrimsRemoved({{_primPath}});
			}
			_isPopulated = populate;
		}

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

			_prim.dataSource = HdOverlayContainerDataSource::New(
				HdRetainedContainerDataSource::New(
					HdPrimvarsSchemaTokens->primvars,
					HdRetainedContainerDataSource::New(
						pxr::HdPrimvarsSchemaTokens->points,
							pxr::HdPrimvarSchema::Builder()
								.SetPrimvarValue(hdPoints)
								.SetRole(hdRole)
								.SetInterpolation(hdInterpolation)
								.Build())),
				_prim.dataSource);

			HdSceneIndexObserver::DirtiedPrimEntries entries;
			HdDataSourceLocator locator(HdPrimvarsSchemaTokens->primvars);
			entries.push_back({_primPath, locator});

			_SendPrimsDirtied(entries);
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

			_prim.dataSource = HdOverlayContainerDataSource::New(
				HdRetainedContainerDataSource::New(
					HdMeshSchemaTokens->mesh,
					HdMeshSchema::Builder()
						.SetTopology(
							pxr::HdMeshTopologySchema::Builder()
								.SetFaceVertexCounts(hdFvc)
								.SetFaceVertexIndices(hdFvi)
								.SetOrientation(hdOrientaion)
								.Build())
						.Build()),
				_prim.dataSource);

			HdSceneIndexObserver::DirtiedPrimEntries entries;
			HdDataSourceLocator locator(HdMeshSchemaTokens->mesh);
			entries.push_back({_primPath, locator});

			_SendPrimsDirtied(entries);
		}

		void SetPrimXform(pxr::GfMatrix4d xform)
		{
			auto hdXform = pxr::HdRetainedTypedSampledDataSource<pxr::GfMatrix4d>::New(
				xform
			);

			auto hdResetTransformStack = pxr::HdRetainedTypedSampledDataSource<bool>::New(
				false
			);

			_prim.dataSource = HdOverlayContainerDataSource::New(
				HdRetainedContainerDataSource::New(
					HdXformSchemaTokens->xform,
					pxr::HdXformSchema::Builder()
							.SetMatrix(hdXform)
							.SetResetXformStack(hdResetTransformStack)
							.Build()),
				_prim.dataSource);
		}

        /**
         * @brief Get the prim at the given path
         *
         * @param primPath the path to a prim
         * @return pxr::HdSceneIndexPrim the hydra prim
         */
        virtual pxr::HdSceneIndexPrim GetPrim(
            const pxr::SdfPath& primPath) const
		{
			if (primPath == _primPath) return _prim;
    		else return {pxr::TfToken(), nullptr};
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
			if (!_isPopulated) return {};
    		if (primPath == pxr::SdfPath::AbsoluteRootPath()) return {_primPath};
   			else return {};
		}

    private:
        pxr::SdfPath _primPath;
        pxr::HdSceneIndexPrim _prim;
        bool _isPopulated;

        /**
         * @brief Create the grid hydra prim
         * 
         * @return pxr::HdSceneIndexPrim the hydra prim of the grid
         */
        pxr::HdSceneIndexPrim _CreateCubePrim()
		{
			// https://github.com/PixarAnimationStudios/OpenUSD/blob/7f5e51901961b4dbbf178a45349431882ba3591f/pxr/imaging/hd/testenv/testHdDataSource.cpp#L190
			pxr::HdSceneIndexPrim prim = pxr::HdSceneIndexPrim(
				{
					pxr::HdPrimTypeTokens->mesh,
					pxr::HdRetainedContainerDataSource::New()
				}
			);

			return prim;
		}
};

PXR_NAMESPACE_CLOSE_SCOPE


//////////////////////////////////////////////////////////////////////////
// Utilities
//////////////////////////////////////////////////////////////////////////

namespace
{
class ScopedTransform
{
	public:
		ScopedTransform( const M44f &transform )
		{
			m_nonIdentity = transform != M44f();
			if( m_nonIdentity )
			{
				glPushMatrix();
				glMultMatrixf( transform.getValue() );
			}
		}

		~ScopedTransform()
		{
			if( m_nonIdentity )
			{
				glPopMatrix();
			}
		}

	private :
		bool m_nonIdentity;
};

template <class... Vs>
bool haveMatchingVisualisations( Visualisation::ColorSpace colorSpace, Visualisation::Scale scale, Visualisation::Category category, const Vs & ... visualisations )
{
	for( auto vs : { visualisations... } )
	{
		for( auto v : vs )
		{
			if( v.colorSpace == colorSpace && v.scale == scale && v.category & category )
			{
				return true;
			}
		}
	}
	return false;
}

template <class... Vs>
void renderMatchingVisualisations( Visualisation::ColorSpace colorSpace, Visualisation::Scale scale, Visualisation::Category category, IECoreGL::State *state, const Vs & ... visualisations )
{
	for( auto vs : { visualisations... } )
	{
		for( auto v : vs )
		{
			if( v.colorSpace == colorSpace && v.scale == scale && v.category & category )
			{
				v.renderable->render( state );
			}
		}
	}
}

template <class... Vs>
void accumulateVisualisationBounds( Box3f &target, Visualisation::Scale scale, Visualisation::Category category, const M44f &transform, const Vs & ... visualisations )
{
	for( auto vs : { visualisations... } )
	{
		for( auto v : vs )
		{
			if( !v.affectsFramingBound || v.scale != scale || !(v.category & category) )
			{
				continue;
			}

			const Box3f b = v.renderable->bound();
			if( !b.isEmpty() )
			{
				target.extendBy( Imath::transform( b, transform ) );
			}
		}
	}
}

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

template<typename T>
T parameter( const IECore::CompoundDataMap &parameters, const IECore::InternedString &name, const T &defaultValue )
{
	IECore::CompoundDataMap::const_iterator it = parameters.find( name );
	if( it == parameters.end() )
	{
		return defaultValue;
	}

	using DataType = IECore::TypedData<T>;
	if( const DataType *d = reportedCast<const DataType>( it->second.get(), "parameter", name ) )
	{
		return d->readable();
	}
	else
	{
		return defaultValue;
	}
}

const IECoreGL::State &selectedSceneState()
{
	static IECoreGL::StatePtr s;
	if( !s )
	{
		s = new IECoreGL::State( false );
		// Turn off wireframe when rendering `ColorSpace::Scene`, because we'll
		// be using it for a selection overlay in `ColorSpace::Display`.
		s->add( new IECoreGL::Primitive::DrawWireframe( false ), /* override = */ true );
	}
	return *s;
}

const IECoreGL::State &selectedCurvesSceneState()
{
	static IECoreGL::StatePtr s;
	if( !s )
	{
		s = new IECoreGL::State( false );
		// Turn off wireframe as for `selectedSceneState()`, but also turn off solid drawing
		// because it also conflicts with the wireframe.
		s->add( new IECoreGL::Primitive::DrawWireframe( false ), /* override = */ true );
		s->add( new IECoreGL::Primitive::DrawSolid( false ), /* override = */ true );
	}
	return *s;
}

const IECoreGL::State &selectedPointsSceneState()
{
	static IECoreGL::StatePtr s;
	if( !s )
	{
		s = new IECoreGL::State( false );
		// See above.
		s->add( new IECoreGL::Primitive::DrawWireframe( false ), /* override = */ true );
		s->add( new IECoreGL::Primitive::DrawSolid( false ), /* override = */ true );
		s->add( new IECoreGL::Primitive::DrawPoints( false ), /* override = */ true );
	}
	return *s;
}

const IECoreGL::State &selectedDisplayState()
{
	static IECoreGL::StatePtr s;
	if( !s )
	{
		s = new IECoreGL::State( false );
		s->add( new IECoreGL::Primitive::DrawPoints( false ), /* override = */ true );
		s->add( new IECoreGL::Primitive::DrawSolid( false ), /* override = */ true );
		s->add( new IECoreGL::Primitive::DrawWireframe( true ), /* override = */ true );
		s->add( new IECoreGL::WireframeColorStateComponent( Color4f( 0.466f, 0.612f, 0.741f, 1.0f ) ), /* override = */ true );
	}
	return *s;
}

const IECoreGL::State &selectionState( const IECoreGL::Renderable *renderable, const IECoreGL::State *currentState, Visualisation::ColorSpace colorSpace )
{
	if( colorSpace == Visualisation::ColorSpace::Display )
	{
		return selectedDisplayState();
	}

	switch( (IECoreGL::TypeId)renderable->typeId() )
	{
		case IECoreGL::PointsPrimitiveTypeId :
			if( static_cast<const IECoreGL::PointsPrimitive *>( renderable )->renderUsesGLPoints( currentState ) ) {
				return selectedPointsSceneState();
			} else {
				return selectedSceneState();
			}
		case IECoreGL::CurvesPrimitiveTypeId :
			if( static_cast<const IECoreGL::CurvesPrimitive *>( renderable )->renderUsesGLLines( currentState ) ) {
				return selectedCurvesSceneState();
			} else {
				return selectedSceneState();
			}
		default :
			return selectedSceneState();
	}
}

} // namespace

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
// OpenGLObject
//////////////////////////////////////////////////////////////////////////

namespace
{

using Edit = std::function<void ()>;
using EditQueue = tbb::concurrent_queue<Edit>;

class OpenGLObject : public IECoreScenePreview::Renderer::ObjectInterface
{

	public :

		OpenGLObject( const std::string &name, const IECore::Object *object, const ConstOpenGLAttributesPtr &attributes, EditQueue &editQueue )
			:	m_objectType( object ? object->typeId() : IECore::NullObjectTypeId ),
				m_attributes( attributes ),
				m_editQueue( editQueue )
		{
			std::cout << "created: " << name << std::endl;
			IECore::StringAlgo::tokenize( name, '/', m_name );

			if( object )
			{
				if( const ObjectVisualiser *visualiser = IECoreGLPreview::ObjectVisualiser::acquire( object->typeId() ) )
				{
					m_objectVisualisations = visualiser->visualise( object );
					m_renderable = nullptr;
				}
				else
				{
					try
					{
						IECore::ConstRunTimeTypedPtr glObject = IECoreGL::CachedConverter::defaultCachedConverter()->convert( object );
						m_renderable = IECore::runTimeCast<const IECoreGL::Renderable>( glObject.get() );
					}
					catch( ... )
					{
						// Leave m_renderable as null
					}
				}
			}
		}

		void transform( const Imath::M44f &transform ) override
		{
			m_editQueue.push( [this, transform]() {
				m_transform = transform;
				m_transformSansScale = sansScalingAndShear( transform, false );
			} );
		}

		void transform( const std::vector<Imath::M44f> &samples, const std::vector<float> &times ) override
		{
			transform( samples.front() );
		}

		bool attributes( const IECoreScenePreview::Renderer::AttributesInterface *attributes ) override
		{
			ConstOpenGLAttributesPtr openGLAttributes = static_cast<const OpenGLAttributes *>( attributes );
			m_editQueue.push( [this, openGLAttributes]() {
				m_attributes = openGLAttributes;
			} );
			return true;
		}

		void link( const IECore::InternedString &type, const IECoreScenePreview::Renderer::ConstObjectSetPtr &objects ) override
		{
		}

		void assignID( uint32_t id ) override
		{
			// The GL renderer provides a more lightweight ID mechanism where
			// IDs are just the index in the object list, and don't need
			// assigning. This is exposed via the `gl:querySelection` command.
			// So we don't implement `assignID()` for now.
			/// \todo Evaluate overhead of the more general ID mechanism, and
			/// consider dropping the custom OpenGL one.
		}

		Box3f transformedBound() const
		{
			Box3f b;

			if( m_renderable )
			{
				const Box3f renderableBound = m_renderable->bound();
				if( !renderableBound.isEmpty() )
				{
					b.extendBy( Imath::transform( renderableBound, m_transform ) );
				}
			}

			Visualisation::Category categories = Visualisation::Category::Generic;
			// Note: We don't have access to selection state here, so we assume it is
			// selected to make sure we consider the frustum if it's enabled.
			if( m_attributes->drawFrustum( true ) )
			{
				categories = Visualisation::Category( categories | Visualisation::Category::Frustum );
			}

			const Visualisations &attrVis = visualisations( *m_attributes );

			accumulateVisualisationBounds( b, Visualisation::Scale::None, categories, m_transformSansScale, attrVis, m_objectVisualisations );
			accumulateVisualisationBounds( b, Visualisation::Scale::Local, categories, m_transform, attrVis, m_objectVisualisations );
			accumulateVisualisationBounds( b, Visualisation::Scale::Visualiser, categories, visualiserTransform( false ), attrVis, m_objectVisualisations );
			accumulateVisualisationBounds( b, Visualisation::Scale::LocalAndVisualiser, categories, visualiserTransform( true ), attrVis, m_objectVisualisations );
			return b;
		}

		const vector<InternedString> &name() const
		{
			return m_name;
		}

		bool selected( const IECore::PathMatcher &selection ) const
		{
			return selection.match( m_name ) & ( PathMatcher::AncestorMatch | PathMatcher::ExactMatch );
		}

		void render( IECoreGL::State *currentState, const IECore::PathMatcher &selection, Visualisation::ColorSpace colorSpace ) const
		{
			const Visualisations &attrVis = visualisations( *m_attributes );
			const bool haveVisualisations = attrVis.size() > 0 || m_objectVisualisations.size() > 0;

			if( !haveVisualisations && !m_renderable )
			{
				return;
			}

			const bool isSelected = selected( selection );

			// In order to minimize z-fighting, we draw non-geometric visualisations
			// first and real geometry last, so that they sit on top. This is
			// still prone to flicker, but seems to provide the best results.

			if( haveVisualisations )
			{
				IECoreGL::State::ScopedBinding selectionScope(
					selectedDisplayState(), *currentState, isSelected && colorSpace == Visualisation::ColorSpace::Display
				);

				Visualisation::Category categories = Visualisation::Category::Generic;
				if( m_attributes->drawFrustum( isSelected ) )
				{
					categories = Visualisation::Category( categories | Visualisation::Category::Frustum );
				}

				if( m_attributes->visualiserScale() > 0.0f )
				{
					if( haveMatchingVisualisations( colorSpace, Visualisation::Scale::Visualiser, categories, attrVis, m_objectVisualisations ) )
					{
						ScopedTransform v( visualiserTransform( false ) );
						renderMatchingVisualisations( colorSpace, Visualisation::Scale::Visualiser, categories, currentState, attrVis, m_objectVisualisations );
					}

					if( haveMatchingVisualisations( colorSpace, Visualisation::Scale::LocalAndVisualiser, categories, attrVis, m_objectVisualisations ) )
					{
						ScopedTransform c( visualiserTransform( true ) );
						renderMatchingVisualisations( colorSpace, Visualisation::Scale::LocalAndVisualiser, categories, currentState, attrVis, m_objectVisualisations );
					}
				}

				if( haveMatchingVisualisations( colorSpace, Visualisation::Scale::None, categories, attrVis, m_objectVisualisations ) )
				{
					ScopedTransform l( m_transformSansScale );
					renderMatchingVisualisations( colorSpace, Visualisation::Scale::None, categories, currentState, attrVis, m_objectVisualisations );
				}

				if( haveMatchingVisualisations( colorSpace, Visualisation::Scale::Local, categories, attrVis, m_objectVisualisations ) )
				{
					ScopedTransform l( m_transform );
					renderMatchingVisualisations( colorSpace, Visualisation::Scale::Local, categories, currentState, attrVis, m_objectVisualisations );
				}
			}

			// Objects are rendered into `ColorSpace::Scene`, with the caveat that selection
			// overlays and additional visualisations are drawn into `ColorSpace::Display`.

			const IECoreGL::State *visualisationState = m_attributes->visualisationState( colorSpace );
			if( m_renderable && ( colorSpace == Visualisation::ColorSpace::Scene || isSelected || visualisationState ) )
			{
				IECoreGL::State::ScopedBinding stateScope( *m_attributes->state(), *currentState );
				std::optional<IECoreGL::State::ScopedBinding> visualisationStateScope;
				if( visualisationState )
				{
					visualisationStateScope.emplace( *visualisationState, *currentState );
				}
				IECoreGL::State::ScopedBinding selectionScope(
					selectionState( m_renderable. get(), currentState, colorSpace ),
					*currentState, isSelected
				);

				ScopedTransform l( m_transform );
				m_renderable->render( currentState );
			}
		}

		IECore::TypeId objectType() const
		{
			return m_objectType;
		}

	protected :

		EditQueue &editQueue()
		{
			return m_editQueue;
		}

		virtual const Visualisations &visualisations( const OpenGLAttributes &attributes ) const
		{
			return attributes.visualisations();
		}

	private :

		// sansScalingAndShear is expensive, so we store that, the other
		// visualiser scaled variants we compute in transformedBound/render
		// to save memory.

		M44f visualiserTransform( bool includeLocal ) const
		{
			M44f t = includeLocal ? m_transform : m_transformSansScale;
			t.scale( V3f( m_attributes->visualiserScale() ) );
			return t;
		}

		IECore::TypeId m_objectType;
		M44f m_transform;
		M44f m_transformSansScale;
		ConstOpenGLAttributesPtr m_attributes;
		IECoreGL::ConstRenderablePtr m_renderable;
		Visualisations m_objectVisualisations;
		vector<InternedString> m_name;
		EditQueue &m_editQueue;

};

IE_CORE_FORWARDDECLARE( OpenGLObject )

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

		~HydraCamera() override
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

class OpenGLLight : public OpenGLObject
{

	public :

		OpenGLLight( const std::string &name, const IECore::Object *light, const ConstOpenGLAttributesPtr &attributes, EditQueue &editQueue )
			:	OpenGLObject( name, light, attributes, editQueue )
		{
		}

	protected :

		const Visualisations &visualisations( const OpenGLAttributes &attributes ) const override
		{
			return attributes.lightVisualisations();
		}

};

IE_CORE_FORWARDDECLARE( OpenGLLight )

class OpenGLLightFilter : public OpenGLObject
{

	public :

		OpenGLLightFilter( const std::string &name, const IECore::Object *object, const ConstOpenGLAttributesPtr &attributes, EditQueue &editQueue )
			:	OpenGLObject( name, object, attributes, editQueue )
		{
		}

	protected :

		const Visualisations &visualisations( const OpenGLAttributes &attributes ) const override
		{
			return attributes.lightFilterVisualisations();
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
			if( renderType == SceneDescription )
			{
				throw IECore::Exception( "Unsupported render type" );
			}

			sceneIndex = pxr::CubeSceneIndex::New();
		}

		~HydraRenderer() override
		{
		}

		IECore::InternedString name() const override
		{
			return "OpenGL";
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
			m_editQueue.push( [ this, result ]() { m_attributes.push_back( result ); } );
			return result;
		}

		ObjectInterfacePtr camera( const std::string &name, const IECoreScene::Camera *camera, const AttributesInterface *attributes ) override
		{
			std::cout << "camera: " << name << std::endl;
			std::cout << "camera attr: " << attributes << std::endl;

			HydraCameraPtr hdCam;
			CameraMap::const_iterator it = m_cameras.find( name );
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
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			OpenGLLightPtr result = new OpenGLLight( name, object, static_cast<const OpenGLAttributes *>( attributes ), m_editQueue );
			m_editQueue.push( [this, result]() { m_objects.push_back( result ); } );
			return result;
		}

		ObjectInterfacePtr lightFilter( const std::string &name, const IECore::Object *object, const AttributesInterface *attributes ) override
		{
			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			OpenGLLightFilterPtr result = new OpenGLLightFilter( name, object, static_cast<const OpenGLAttributes *>( attributes ), m_editQueue );
			m_editQueue.push( [this, result]() { m_objects.push_back( result ); } );
			return result;
		}

		Renderer::ObjectInterfacePtr object( const std::string &name, const IECore::Object *object, const AttributesInterface *attributes ) override
		{
			if (object->typeId() == MeshPrimitive::staticTypeId())
			{
				auto *mesh = dynamic_cast<const IECoreScene::MeshPrimitive*>(object);
				std::cout << mesh->interpolation() << std::endl;
				std::cout << mesh->maxVerticesPerFace() << std::endl;

				auto &verticesPerFace = mesh->verticesPerFace()->readable();
				auto &vertexIds = mesh->vertexIds()->readable();

				const V3fVectorData *p = mesh->variableData<V3fVectorData>( "P", PrimitiveVariable::Vertex );
				const vector<Imath::V3f> &points = p->readable();
				pxr::VtVec3fArray pts;
				for( auto pt: points)
					pts.push_back(pxr::GfVec3f( IECoreUSD::DataAlgo::toUSD( pt ) ));

				std::cout << pxr::VtIntArray( verticesPerFace.begin(), verticesPerFace.end() ) << std::endl;
				std::cout << pxr::VtIntArray( vertexIds.begin(), vertexIds.end() ) << std::endl;
				// std::cout << IECoreUSD::DataAlgo::toUSD( points ) << std::endl;

				if( name == "/sphere")
				{
					sceneIndex->SetPrimTopology(pxr::VtIntArray( verticesPerFace.begin(), verticesPerFace.end() ), pxr::VtIntArray( vertexIds.begin(), vertexIds.end() ));
					sceneIndex->SetPrimPoints(pts);
				}
			}
			std::cout << "object2: " << name << " " << object->typeId() << " " << MeshPrimitive::staticTypeId() << std::endl;
			std::cout << "m_renderObjects: " << m_renderObjects << std::endl;

			if( !m_renderObjects && !runTimeCast<const IECoreScenePreview::Placeholder>( object ) )
			{
				return nullptr;
			}

			std::cout << "object3: " << name << std::endl;

			IECore::MessageHandler::Scope s( m_messageHandler.get() );

			OpenGLObjectPtr result = new OpenGLObject( name, object, static_cast<const OpenGLAttributes *>( attributes ), m_editQueue );
			m_editQueue.push( [this, result]() { m_objects.push_back( result ); } );
			return result;
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
			const string colorSpaceString = parameter<string>( parameters, "colorSpace", "scene" );
			const Visualisation::ColorSpace colorSpace = colorSpaceString == "scene" ? Visualisation::ColorSpace::Scene : Visualisation::ColorSpace::Display;

			// processQueue();
			// removeDeletedObjects();
			// CachedConverter::defaultCachedConverter()->clearUnused();

			GLint prevProgram;
			glGetIntegerv( GL_CURRENT_PROGRAM, &prevProgram );
			glPushAttrib( GL_ALL_ATTRIB_BITS );

				// State::bindBaseState();
				// State *state = baseState();
				// state->bind();

			

				// if( IECoreGL::Selector *selector = IECoreGL::Selector::currentSelector() )
				// {
				// 	// IECoreGL expects us to bind `selector->baseState()` here, so the
				// 	// selector can control a few specific parts of the state.
				// 	// That overrides _all_ of our own state though, including things that
				// 	// are crucial to accurate selection because they change the size of
				// 	// primitives on screen. So we need to bind the selection state and then
				// 	// rebind the crucial bits of our state back on top of it.
				// 	/// \todo Change IECoreGL::Selector so it provides a partial state object
				// 	/// containing only the things it needs to change.
				// 	IECoreGL::StatePtr shapeState = new IECoreGL::State( /* complete = */ false );
				// 	shapeState->add( state->get<IECoreGL::Primitive::DrawWireframe>() );
				// 	shapeState->add( state->get<IECoreGL::Primitive::DrawSolid>() );
				// 	shapeState->add( state->get<IECoreGL::Primitive::DrawOutline>() );
				// 	shapeState->add( state->get<IECoreGL::Primitive::DrawPoints>() );
				// 	shapeState->add( state->get<IECoreGL::PointsPrimitive::UseGLPoints>() );
				// 	shapeState->add( state->get<IECoreGL::PointsPrimitive::GLPointWidth>() );
				// 	shapeState->add( state->get<IECoreGL::CurvesPrimitive::UseGLLines>() );
				// 	shapeState->add( state->get<IECoreGL::CurvesPrimitive::IgnoreBasis>() );
				// 	shapeState->add( state->get<IECoreGL::CurvesPrimitive::GLLineWidth>() );
				// 	IECoreGL::State::ScopedBinding selectorStateBinding(
				// 		*selector->baseState(), const_cast<IECoreGL::State &>( *state )
				// 	);
				// 	IECoreGL::State::ScopedBinding shapeStateBinding(
				// 		*shapeState, const_cast<IECoreGL::State &>( *state )
				// 	);
				// 	renderObjects( state, colorSpace );
				// }
				// else
				// {
				// 	renderObjects( state, colorSpace );
				// }

			
			if(!hgi){
				std::cout << "init ghi" << std::endl;

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
				const pxr::SdfPathVector paths = {pxr::SdfPath("/stageSceneIndex/Cube")};

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
				CameraMap::const_iterator it = m_cameras.find( m_camera );
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

			// pxr::VtValue aov;
			// pxr::HgiTextureHandle aovTexture;

			// if (engine.GetTaskContextData(pxr::HdAovTokens->color, &aov)) {
			// 	if (aov.IsHolding<pxr::HgiTextureHandle>()) {
			// 		aovTexture = aov.Get<pxr::HgiTextureHandle>();
			// 	}
			// }

			// uint32_t framebuffer = 0;
			// pxr::HgiInterop interop;
			// interop.TransferToApp(hgi.get(), aovTexture, pxr::HgiTextureHandle(),
			// 					pxr::HgiTokens->OpenGL, pxr::VtValue(framebuffer),
			// 					pxr::GfVec4i(0, 0, WIDTH, HEIGHT));

			// drawTarget->WriteToFile("color", "/Users/raphaeljouretz/Desktop/test.png");
			// drawTarget->Unbind();


			glPopAttrib();
			glUseProgram( prevProgram );
		}

		void renderBatch()
		{
			
		}

		void processQueue()
		{
			Edit edit;
			while( m_editQueue.try_pop( edit ) )
			{
				edit();
			}
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
			std::cout << "DELETE" << std::endl;
			for( auto it = m_cameras.begin(); it != m_cameras.end(); )
			{
				// Cameras are referenced by both m_cameras and m_objects
				if( it->second->refCount() == 2 )
				{
					it = m_cameras.erase( it );
				}
				else
				{
					++it;
				}
			}

			m_objects.erase(
				remove_if(
					m_objects.begin(),
					m_objects.end(),
					[]( const OpenGLObjectPtr &o ) { return o->refCount() == 1; }
				),
				m_objects.end()
			);

			m_attributes.erase(
				remove_if(
					m_attributes.begin(),
					m_attributes.end(),
					[]( const OpenGLAttributesPtr &a ) { return a->refCount() == 1; }
				),
				m_attributes.end()
			);
		}

		void renderObjects( IECoreGL::State *currentState, Visualisation::ColorSpace colorSpace )
		{
			IECoreGL::Selector *selector = IECoreGL::Selector::currentSelector();

			GLuint i = 1;
			for( const auto &o : m_objects )
			{
				if( selector )
				{
					selector->loadName( i++ );
				}
				o->render( currentState, m_selection, colorSpace );
			}
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
			const bool selected = parameter<bool>( parameters, "selection", false );

			const PathMatcher omitted = parameter<PathMatcher>( parameters, "omitted", PathMatcher() );
			const bool omittedEmpty = omitted.isEmpty();

			processQueue();
			removeDeletedObjects();

			Box3f result;
			for( const auto &o : m_objects )
			{
				if(
					( selected && !o->selected( m_selection ) ) ||
					( !omittedEmpty && ( omitted.match( o->name() ) & ( PathMatcher::AncestorMatch | PathMatcher::ExactMatch ) ) )
				)
				{
					continue;
				}

				result.extendBy( o->transformedBound() );
			}
			return new Box3fData( result );
		}

		DataPtr querySelectedObjects( const CompoundDataMap &parameters )
		{
			ConstUIntVectorDataPtr names;
			CompoundDataMap::const_iterator it = parameters.find( "selection" );
			if( it != parameters.end() )
			{
				names = runTimeCast<const UIntVectorData>( it->second );
			}
			if( !names )
			{
				throw InvalidArgumentException( "Expected UIntVectorData \"selection\" parameter" );
			}

			vector<IECore::TypeId> maskTypeIds;
			it = parameters.find( "mask" );
			if( it != parameters.end() )
			{
				if( ConstStringVectorDataPtr typeNames = runTimeCast<const StringVectorData>( it->second ) )
				{
					for( const auto &n : typeNames->readable() )
					{
						maskTypeIds.push_back( RunTimeTyped::typeIdFromTypeName( n.c_str() ) );
					}
				}
				else
				{
					throw InvalidArgumentException( "Expected StringVectorData for \"mask\" parameter" );
				}
			}
			else
			{
				maskTypeIds.push_back( IECore::ObjectTypeId );
			}

			PathMatcher result;
			for( auto i : names->readable() )
			{
				const OpenGLObject *o = m_objects[i-1].get();
				for( auto t : maskTypeIds )
				{
					if( t == o->objectType() || RunTimeTyped::inheritsFrom( o->objectType(), t ) )
					{
						result.addPath( o->name() );
						break;
					}
				}
			}

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

		// Queue used to pass edits from background threads to the render thread.
		EditQueue m_editQueue;

		// Render state. Updated on the render thread by processing Edits
		// from m_editQueue.

		unordered_map<InternedString, ConstOutputPtr> m_outputs;
		using CameraMap = std::unordered_map<string, HydraCameraPtr>;
		CameraMap m_cameras;
		HydraCameraPtr m_cam = nullptr;

		using OpenGLObjectVector = std::vector<OpenGLObjectPtr>;
		OpenGLObjectVector m_objects;

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

		pxr::CubeSceneIndexRefPtr sceneIndex;

};

IECoreScenePreview::Renderer::TypeDescription<HydraRenderer> HydraRenderer::g_typeDescription( "Hydra" );

} // namespace
