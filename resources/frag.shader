#version 410 core

#define PI 3.1415926
#define TwoPI (2.0 * PI)

vec2 envMapEquirect(vec3 wcNormal, float flipEnvMap) {
  //I assume envMap texture has been flipped the WebGL way (pixel 0,0 is a the bottom)
  //therefore we flip wcNorma.y as acos(1) = 0
  float phi = acos(wcNormal.y);
  float theta = atan(flipEnvMap * wcNormal.x, wcNormal.z)*3 + PI;
  return vec2(theta / TwoPI, phi / PI);
}

vec2 envMapEquirect(vec3 wcNormal) {
    //-1.0 for left handed coordinate system oriented texture (usual case)
    return envMapEquirect(wcNormal, -1.0);
}

vec3 tonemapFilmic(vec3 color) {
    vec3 x = max(vec3(0.0), color - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}


float uIor= 1.60;

float saturate(float f) {
    return clamp(f, 0.0, 1.0);
}

vec3 toGamma(vec3 color) {
	return pow(color, vec3(1.0 / 2.2));
}

vec3 toLinear(vec3 color) {
	return pow(color, vec3(2.2));
}


uniform sampler2D Albedo;
uniform sampler2D Roughness;
uniform sampler2D Normal;
uniform sampler2D Metallic;

uniform sampler2D hdrTexture;

in vec3 v_vertex;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec2 v_uv;
in vec3 v_worldPos;

layout (location = 0) out vec4 out_color;

vec3 getAlbedo() {
    return toLinear(texture(Albedo, v_uv).rgb);
}


float getRoughness() {
    return texture(Roughness, v_uv).r;
}


float getMetalness() {
    return texture(Metallic, v_uv).r;
}

void calculateBasis(out vec3 tangent, out vec3 bitangent, in vec3 normal)
{
	bitangent = vec3(0.0, 1.0, 0.0);

	float normalDotUp = dot(normal, bitangent);

	if (normalDotUp == 1.0)
	{
		bitangent = vec3(0.0, 0.0, -1.0);
	}
	else if (normalDotUp == -1.0)
	{
		bitangent = vec3(0.0, 0.0, 1.0);
	}
	
	tangent = cross(bitangent, normal);	
	bitangent = cross(normal, tangent);
} 

vec3 getNormal(float height_factor) {
	/*
    vec3 normalRGB = texture2D(Normal, vTexCord0).rgb;
    vec3 normalMap = normalRGB * 2.0 - 1.0;

    normalMap.y *= -1.0;

    vec3 N = normalize(vNormalView);
    vec3 V = normalize(vEyeDirView);

    vec3 normalView = perturb(normalMap, N, V, vTexCord0);
    vec3 normalWorld = vec3(uInverseViewMatrix * vec4(normalView, 0.0));
    return normalWorld;

	*/

	vec3 normalmap = texture(Normal, v_uv).rgb;
    normalmap = normalize(normalmap * 2.0 - 1.0);
	normalmap.y *= -1.0;

	normalmap= mix(vec3(0,0,1),normalmap,height_factor);

	// Tangent, Bitangent and Normal are in world space.
	vec3 tangent = normalize(v_tangent);
	vec3 bitangent = normalize(v_bitangent);
	vec3 normal = normalize(v_normal);

	mat3 basis = mat3(tangent, bitangent, normal);
	
	normal = normalize(basis*normalmap);
	return normal;

}





vec3 getIrradiance(vec3 eyeDirWorld, vec3 normalWorld) {
    float maxMipMapLevel = 7.0; //TODO: const
    vec3 reflectionWorld = reflect(eyeDirWorld, normalWorld);
    vec2 R = envMapEquirect(normalWorld);
    return textureLod(hdrTexture, R,8).rgb;
}

vec3 EnvBRDFApprox( vec3 SpecularColor, float Roughness, float NoV ) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022 );
    const vec4 c1 = vec4( 1.0, 0.0425, 1.04, -0.04 );
    vec4 r = Roughness * c0 + c1;
    float a004 = min( r.x * r.x, exp2( -9.28 * NoV ) ) * r.x + r.y;
    vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;
    return SpecularColor * AB.x + AB.y;
}

vec3 getPrefilteredReflection(vec3 eyeDirWorld, vec3 normalWorld, float roughness) {
    float maxMipMapLevel = 8.0; //TODO: const
    vec3 reflectionWorld = reflect(eyeDirWorld, normalWorld);
    vec2 R = envMapEquirect(reflectionWorld);
    float lod = roughness * maxMipMapLevel;
    float upLod = floor(lod);
    float downLod = ceil(lod);
    vec3 a = toLinear(textureLod(hdrTexture, R, upLod).rgb);
    vec3 b = toLinear(textureLod(hdrTexture, R, downLod).rgb);

    return mix(a, b, lod - upLod);
}

uniform vec3 cameraPos;
uniform float height_factor;

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x) {
	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
 }
 
 //Based on Filmic Tonemapping Operators http://filmicgames.com/archives/75
 vec3 tonemapUncharted2(vec3 color) {
	 float ExposureBias = 2.0;
	 vec3 curr = Uncharted2Tonemap(ExposureBias * color);
 
	 vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(W));
	 return curr * whiteScale;
 }



void main() {

    vec3 normalWorld = getNormal(height_factor);
    vec3 eyeDirWorld = normalize(v_worldPos-cameraPos);

    vec3 albedo = getAlbedo();
    float roughness = pow(getRoughness(),1/2.2);
    float metalness = getMetalness();



    vec3 irradianceColor = getIrradiance(eyeDirWorld, normalWorld)*4;
    vec3 reflectionColor = getPrefilteredReflection(eyeDirWorld, normalWorld, roughness)*4;

    vec3 F0 = vec3(abs((1.0 - uIor) / (1.0 + uIor)));
    F0 = F0 * F0;
    F0 = mix(F0, albedo, metalness);

    float NdotV = saturate( dot( normalWorld, eyeDirWorld ) );
    vec3 reflectance = EnvBRDFApprox( F0, roughness, NdotV );

    vec3 diffuseColor = albedo * (1.0 - metalness);
	vec3 color = diffuseColor * irradianceColor + reflectionColor * reflectance;

    color = tonemapUncharted2(color*3);
    color = toGamma(color);
	out_color = vec4(color, 1.0);

}