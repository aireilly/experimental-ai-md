# Deploying {prodname-short} in {ocp-product-title}

You can deploy Red Hat AI Inference Server in OpenShift Container Platform clusters with supported AI accelerators that have full access to the internet.

<dl><dt><strong>📌 NOTE</strong></dt><dd>

Install the NVIDIA GPU Operator or AMD GPU Operator as appropriate for the underlying host AI accelerators that are available in the cluster.
</dd></dl>

# Installing the Node Feature Discovery Operator

Install the Node Feature Discovery Operator so that the cluster can use the AI accelerators that are available in the cluster.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.

1. Create the `Namespace` CR for the Node Feature Discovery Operator:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: v1
    kind: Namespace
    metadata:
      name: openshift-nfd
      labels:
        name: openshift-nfd
        openshift.io/cluster-monitoring: "true"
    EOF
    ```
2. Create the `OperatorGroup` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1
    kind: OperatorGroup
    metadata:
      generateName: openshift-nfd-
      name: openshift-nfd
      namespace: openshift-nfd
    spec:
      targetNamespaces:
      - openshift-nfd
    EOF
    ```
3. Create the `Subscription` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1alpha1
    kind: Subscription
    metadata:
      name: nfd
      namespace: openshift-nfd
    spec:
      channel: "stable"
      installPlanApproval: Automatic
      name: nfd
      source: redhat-operators
      sourceNamespace: openshift-marketplace
    EOF
    ```

Verify that the Node Feature Discovery Operator deployment is successful by running the following command:

```terminal
$ oc get pods -n openshift-nfd
```

**Example output**

```terminal
NAME                                      READY   STATUS    RESTARTS   AGE
nfd-controller-manager-7f86ccfb58-vgr4x   2/2     Running   0          10m
```

* [Installing the Node Feature Discovery Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/specialized_hardware_and_driver_enablement/psap-node-feature-discovery-operator#installing-the-node-feature-discovery-operator_psap-node-feature-discovery-operator)

# Installing the NVIDIA GPU Operator

Install the NVIDIA GPU Operator to use the underlying NVIDIA CUDA AI accelerators that are available in the cluster.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.
* You have installed the Node Feature Discovery Operator.

1. Create the `Namespace` CR for the NVIDIA GPU Operator:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: v1
    kind: Namespace
    metadata:
      name: nvidia-gpu-operator
    EOF
    ```
2. Create the `OperatorGroup` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1
    kind: OperatorGroup
    metadata:
      name: gpu-operator-certified
      namespace: nvidia-gpu-operator
    spec:
     targetNamespaces:
     - nvidia-gpu-operator
    EOF
    ```
3. Create the `Subscription` CR:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: operators.coreos.com/v1alpha1
    kind: Subscription
    metadata:
      name: gpu-operator-certified
      namespace: nvidia-gpu-operator
    spec:
      channel: "stable"
      installPlanApproval: Manual
      name: gpu-operator-certified
      source: certified-operators
      sourceNamespace: openshift-marketplace
    EOF
    ```

Verify that the NVIDIA GPU Operator deployment is successful by running the following command:

```terminal
$ oc get pods -n nvidia-gpu-operator
```

**Example output**

```terminal
NAME                                                  READY   STATUS     RESTARTS   AGE
gpu-feature-discovery-c2rfm                           1/1     Running    0          6m28s
gpu-operator-84b7f5bcb9-vqds7                         1/1     Running    0          39m
nvidia-container-toolkit-daemonset-pgcrf              1/1     Running    0          6m28s
nvidia-cuda-validator-p8gv2                           0/1     Completed  0          99s
nvidia-dcgm-exporter-kv6k8                            1/1     Running    0          6m28s
nvidia-dcgm-tpsps                                     1/1     Running    0          6m28s
nvidia-device-plugin-daemonset-gbn55                  1/1     Running    0          6m28s
nvidia-device-plugin-validator-z7ltr                  0/1     Completed  0          82s
nvidia-driver-daemonset-410.84.202203290245-0-xxgdv   2/2     Running    0          6m28s
nvidia-node-status-exporter-snmsm                     1/1     Running    0          6m28s
nvidia-operator-validator-6pfk6                       1/1     Running    0          6m28s
```

* [Installing the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/mirror-gpu-ocp-disconnected.html#deploy-gpu-operators-in-a-disconnected-or-airgapped-environment)

# Installing the AMD GPU Operator

Install the AMD GPU Operator to use the underlying AMD ROCm AI accelerators that are available in the cluster.

Installing the AMD GPU Operator is a multi-step procedure that requires installing the Node Feature Discovery Operator, the Kernel Module Management Operator (KMM), and then the AMD GPU Operator through the OpenShift OperatorHub.

<dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

The AMD GPU Operator is only supported in clusters with full access to the internet, not in disconnected environments.
This is because the Operator builds the driver inside the cluster which requires full internet access.
</dd></dl>

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.
* You have installed the following Operators in the cluster:

    **Required Operators**

    |     |     |
    | --- | --- |
    | Operator | Description |
    | Service CA Operator | Issues TLS serving certificates for Service objects. Required for certificate signing and authentication between the `kube-apiserver` and the KMM webhook server. |
    | Operator Lifecycle Manager (OLM) | Manages Operator installation and lifecycle maintenance. |
    | Machine Config Operator | Manages the operating system configuration of worker and control-plane nodes. Required for configuring the kernel blacklist for the amdgpu driver. |
    | Cluster Image Registry Operator | The Cluster Image Registry Operator (CIRO) manages the internal container image registry that OpenShift Container Platform clusters use to store and serve container images. Required for driver image building and storage in the cluster. |

1. Create the `Namespace` CR for the AMD GPU Operator Operator:

    ```yaml
    oc apply -f - <<EOF
    apiVersion: v1
    kind: Namespace
    metadata:
      name: openshift-amd-gpu
      labels:
        name: openshift-amd-gpu
        openshift.io/cluster-monitoring: "true"
    EOF
    ```
2. Verify that the Service CA Operator is operational. Run the following command:

    ```terminal
    $ oc get pods -A | grep service-ca
    ```

    **Example output**

    ```terminal
    openshift-service-ca-operator   service-ca-operator-7cfd997ddf-llhdg    1/1    Running    7    35d
    openshift-service-ca            service-ca-8675b766d5-vz8gg             1/1    Running    6    35d
    ```
3. Verify that the Machine Config Operator is operational:

    ```terminal
    $ oc get pods -A | grep machine-config-daemon
    ```

    **Example output**

    ```terminal
    openshift-machine-config-operator   machine-config-daemon-sdsjj   2/2    Running    10   35d
    openshift-machine-config-operator   machine-config-daemon-xc6rm   2/2    Running    0    2d21h
    ```
4. Verify that the Cluster Image Registry Operator is operational:

    ```terminal
    $ oc get pods -n openshift-image-registry
    ```

    **Example output**

    ```terminal
    NAME                                               READY   STATUS      RESTARTS   AGE
    cluster-image-registry-operator-58f9dc9976-czt2w   1/1     Running     5          35d
    image-pruner-29259360-2tdrk                        0/1     Completed   0          2d8h
    image-pruner-29260800-v9lkc                        0/1     Completed   0          32h
    image-pruner-29262240-swcmb                        0/1     Completed   0          8h
    image-registry-7b67584cd-sdxpk                     1/1     Running     10         35d
    node-ca-d2kzl                                      1/1     Running     0          2d21h
    node-ca-xxzrw                                      1/1     Running     5          35d
    ```
5. Optional: If you plan to build driver images in the cluster, you must enable the OpenShift internal registry. Run the following commands:
    1. Verify current registry status:

        ```terminal
        $ oc get pods -n openshift-image-registry
        ```

        ```terminal
        NAME                                               READY   STATUS      RESTARTS   AGE
        #...
        image-registry-7b67584cd-sdxpk                     1/1     Running     10         36d
        ```
    2. Configure the registry storage. The following example patches an `emptyDir` ephemeral volume in the cluster. Run the following command:

        ```terminal
        $ oc patch configs.imageregistry.operator.openshift.io cluster --type merge \
          --patch '{"spec":{"storage":{"emptyDir":{}}}}'
        ```
    3. Enable the registry:

        ```terminal
        $ oc patch configs.imageregistry.operator.openshift.io cluster --type merge \
          --patch '{"spec":{"managementState":"Managed"}}'
        ```
6. Install the Node Feature Discovery (NFD) Operator.
See [Installing the Node Feature Discovery Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/4.19/html-single/specialized_hardware_and_driver_enablement/#installing-the-node-feature-discovery-operator_node-feature-discovery-operator).
7. Install the Kernel Module Management (KMM) Operator.
See [Installing the Kernel Module Management Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html-single/specialized_hardware_and_driver_enablement/index#kmm-install_kernel-module-management-operator).
8. Configure node feature discovery for the AMD AI accelerator:
    1. Create a `NodeFeatureDiscovery` (`NFD`) custom resource (CR) to detect AMD GPU hardware. For example:

        ```yaml
        apiVersion: nfd.openshift.io/v1
        kind: NodeFeatureDiscovery
        metadata:
          name: amd-gpu-operator-nfd-instance
          namespace: openshift-nfd
        spec:
          workerConfig:
            configData: |
              core:
                sleepInterval: 60s
              sources:
                pci:
                  deviceClassWhitelist:
                    - "0200"
                    - "03"
                    - "12"
                  deviceLabelFields:
                    - "vendor"
                    - "device"
                custom:
                - name: amd-gpu
                  labels:
                    feature.node.kubernetes.io/amd-gpu: "true"
                  matchAny:
                    - matchFeatures:
                        - feature: pci.device
                          matchExpressions:
                            vendor: {op: In, value: ["1002"]}
                            device: {op: In, value: [
                              "740f", # MI210
                            ]}
                - name: amd-vgpu
                  labels:
                    feature.node.kubernetes.io/amd-vgpu: "true"
                  matchAny:
                    - matchFeatures:
                        - feature: pci.device
                          matchExpressions:
                            vendor: {op: In, value: ["1002"]}
                            device: {op: In, value: [
                              "74b5", # MI300X VF
                            ]}
        ```

        <dl><dt><strong>📌 NOTE</strong></dt><dd>

        Depending on your specific cluster deployment, you might require a `NodeFeatureDiscovery` or `NodeFeatureRule` CR.
        For example, the cluster might already have the `NodeFeatureDiscovery` resource deployed and you don’t want to change it.
        For more information, see [Create Node Feature Discovery Rule](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/installation/openshift-olm.html#create-node-feature-discovery-rule).
        </dd></dl>
9. Create a `MachineConfig` CR to add the out-of-tree `amdgpu` kernel module to the modprobe blacklist. For example:

    ```yaml
    apiVersion: machineconfiguration.openshift.io/v1
    kind: MachineConfig
    metadata:
      labels:
        machineconfiguration.openshift.io/role: worker ①
      name: amdgpu-module-blacklist
    spec:
      config:
        ignition:
          version: 3.2.0
        storage:
          files:
            - path: "/etc/modprobe.d/amdgpu-blacklist.conf"
              mode: 420
              overwrite: true
              contents:
                source: "data:text/plain;base64,YmxhY2tsaXN0IGFtZGdwdQo="
    ```
    1. Set `machineconfiguration.openshift.io/role: master` for single-node OpenShift clusters.

    <dl><dt><strong>❗ IMPORTANT</strong></dt><dd>

    The Machine Config Operator automatically reboots selected nodes after you apply the `MachineConfig` CR.
    </dd></dl>
10. Create the `DeviceConfig` CR to start the AMD AI accelerator driver installation. For example:

    ```yaml
    apiVersion: amd.com/v1alpha1
    kind: DeviceConfig
    metadata:
      name: driver-cr
      namespace: openshift-amd-gpu
    spec:
      driver:
        enable: true
        image: image-registry.openshift-image-registry.svc:5000/$MOD_NAMESPACE/amdgpu_kmod ①
        version: 6.2.2
      selector:
        "feature.node.kubernetes.io/amd-gpu": "true"
    ```
    1. By default, you do not need to configure a value for the `image` field. The default value is shown.

    After you apply the `DeviceConfig` CR, the AMD GPU Operator collects the worker node system specifications, builds or retrieve the appropriate driver image, uses KMM to deploy the driver, and finally deploys the ROCM device plugin and node labeller.

1. Verify that the KMM worker pods are running:

    ```terminal
    $ oc get pods -n openshift-kmm
    ```

    **Example output**

    ```terminal
    NAME                                       READY   STATUS    RESTARTS         AGE
    kmm-operator-controller-774c7ccff6-hr76v   1/1     Running   30 (2d23h ago)   35d
    kmm-operator-webhook-76d7b9555-ltmps       1/1     Running   5                35d
    ```
2. Check device plugin and labeller status:

    ```terminal
    $ oc -n openshift-amd-gpu get pods
    ```

    **Example output**

    ```terminal
    NAME                                                   READY   STATUS    RESTARTS        AGE
    amd-gpu-operator-controller-manager-59dd964777-zw4bg   1/1     Running   8 (2d23h ago)   9d
    test-deviceconfig-device-plugin-kbrp7                  1/1     Running   0               2d
    test-deviceconfig-metrics-exporter-k5v4x               1/1     Running   0               2d
    test-deviceconfig-node-labeller-fqz7x                  1/1     Running   0               2d
    ```
3. Confirm that GPU resource labels are applied to the nodes:

    ```terminal
    $ oc get node -o json | grep amd.com
    ```

    **Example output**

    ```terminal
    "amd.com/gpu.cu-count": "304",
    "amd.com/gpu.device-id": "74b5",
    "amd.com/gpu.driver-version": "6.12.12",
    "amd.com/gpu.family": "AI",
    "amd.com/gpu.simd-count": "1216",
    "amd.com/gpu.vram": "191G",
    "beta.amd.com/gpu.cu-count": "304",
    "beta.amd.com/gpu.cu-count.304": "8",
    "beta.amd.com/gpu.device-id": "74b5",
    "beta.amd.com/gpu.device-id.74b5": "8",
    "beta.amd.com/gpu.family": "AI",
    "beta.amd.com/gpu.family.AI": "8",
    "beta.amd.com/gpu.simd-count": "1216",
    "beta.amd.com/gpu.simd-count.1216": "8",
    "beta.amd.com/gpu.vram": "191G",
    "beta.amd.com/gpu.vram.191G": "8",
    "amd.com/gpu": "8",
    "amd.com/gpu": "8",
    ```

* [Installing the Node Feature Discovery Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/4.19/html-single/specialized_hardware_and_driver_enablement/#installing-the-node-feature-discovery-operator_node-feature-discovery-operator)
* [Installing the Kernel Module Management Operator](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html-single/specialized_hardware_and_driver_enablement/index#kmm-install_kernel-module-management-operator)
* [Installing the AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/installation/openshift-olm.html)

# Deploying Red Hat AI Inference Server and inference serving the model

Deploy a language model with OpenShift Container Platform by configuring secrets, persistent storage, and a deployment custom resource (CR) that pulls the model from Hugging Face and uses Red Hat AI Inference Server to inference serve the model.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.
* You have installed NFD and the required GPU Operator for your underlying AI accelerator hardware.

1. Create the `Secret` custom resource (CR) for the Hugging Face token.
The cluster uses the `Secret` CR to pull models from Hugging Face.
    1. Set the `HF_TOKEN` variable using the token you set in [Hugging Face](https://huggingface.co/settings/tokens).

        ```terminal
        $ HF_TOKEN=<your_huggingface_token>
        ```
    2. Set the cluster namespace to match where you deployed the Red Hat AI Inference Server image, for example:

        ```terminal
        $ NAMESPACE=rhaiis-namespace
        ```
    3. Create the `Secret` CR in the cluster:

        ```terminal
        $ oc create secret generic hf-secret --from-literal=HF_TOKEN=$HF_TOKEN -n $NAMESPACE
        ```
2. Create the Docker secret so that the cluster can download the Red Hat AI Inference Server image from the container registry. For example, to create a `Secret` CR that contains the contents of your local `~/.docker/config.json` file, run the following command:

    ```terminal
    oc create secret generic docker-secret --from-file=.dockercfg=$HOME/.docker/config.json --type=kubernetes.io/dockercfg -n rhaiis-namespace
    ```
3. Create a `PersistentVolumeClaim` (`PVC`) custom resource (CR) and apply it in the cluster.
The following example `PVC` CR uses a default IBM VPC Block persistence volume.
You use the `PVC` as the location where you store the models that you download.

    ```yaml
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: model-cache
      namespace: aireilly-rhaiis
    spec:
      accessModes:
        - ReadWriteOnce
      resources:
        requests:
          storage: 20Gi
      storageClassName: ibmc-vpc-block-10iops-tier
    ```

    <dl><dt><strong>📌 NOTE</strong></dt><dd>

    Configuring cluster storage to meet your requirements is outside the scope of this procedure.
    For more detailed information, see [Configuring persistent storage](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/storage/configuring-persistent-storage).
    </dd></dl>
4. Create a `Deployment` custom resource (CR) that pulls the model from Hugging Face and deploys the Red Hat AI Inference Server container.
Reference the following example `Deployment` CR, which uses AI Inference Server to serve a Granite model on a CUDA accelerator.

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: granite
      namespace: rhaiis-namespace ①
      labels:
        app: granite
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: granite
      template:
        metadata:
          labels:
            app: granite
        spec:
          imagePullSecrets:
            - name: docker-secret
          volumes:
            - name: model-volume
              persistentVolumeClaim:
                claimName: model-cache ②
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            - name: oci-auth
              secret:
                secretName: docker-secret
                items:
                  - key: .dockercfg
                    path: config.json
          serviceAccountName: default
          initContainers: ③
            - name: fetch-model
              image: ghcr.io/oras-project/oras:v1.2.0
              env:
                - name: DOCKER_CONFIG
                  value: /auth
              command: ["/bin/sh","-c"]
              args:
                - |
                  set -e
                  # Only pull if /model is empty
                  if [ -z "$(ls -A /model)" ]; then
                    echo "Pulling model..."
                    oras pull registry.redhat.io/rhelai1/granite-3-1-8b-instruct-quantized-w8a8:1.5 \
                      --output /model \
                  else
                    echo "Model already present, skipping model pull"
                  fi
              volumeMounts:
                - name: model-volume
                  mountPath: /model
                - name: oci-auth
                  mountPath: /auth
                  readOnly: true
          containers:
            - name: granite
              image: 'registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:a6645a8e8d7928dce59542c362caf11eca94bb1b427390e78f0f8a87912041cd'
              imagePullPolicy: IfNotPresent
              env:
                - name: VLLM_SERVER_DEV_MODE
                  value: '1'
              command:
                - python
                - '-m'
                - vllm.entrypoints.openai.api_server
              args:
                - '--port=8000'
                - '--model=/model'
                - '--served-model-name=granite-3-1-8b-instruct-quantized-w8a8'
                - '--tensor-parallel-size=1'
              resources:
                limits:
                  cpu: '10'
                  nvidia.com/gpu: '1'
                  memory: 16Gi
                requests:
                  cpu: '2'
                  memory: 6Gi
                  nvidia.com/gpu: '1'
              volumeMounts:
                - name: model-volume
                  mountPath: /model
                - name: shm
                  mountPath: /dev/shm ④
          restartPolicy: Always
    ```
    1. `metadata.namespace` must match the namespace where you configure the Hugging Face `Secret` CR.
    2. `spec.template.spec.volumes.persistentVolumeClaim.claimName` must match the name of the `PVC` that you created.
    3. This example deployment uses a simple `initContainers` configuration that runs before the main app container to download the required model from Hugging Face.
    The model pull step is skipped if the model directory has already been populated, for example, from a previous deployment.
    4. The `/dev/shm` volume mount is required by the NVIDIA Collective Communications Library (NCCL).
    Tensor parallel vLLM deployments fail when the `/dev/shm` volume mount is not set.
        1. Increase the deployment replica count to the required number. For example, run the following command:

            ```terminal
            oc scale deployment granite -n rhaiis-namespace --replicas=1
            ```
        2. Optional: Watch the deployment and ensure that it succeeds:

            ```terminal
            $ oc get deployment -n rhaiis-namespace --watch
            ```

            **Example output**

            ```terminal
            NAME      READY   UP-TO-DATE   AVAILABLE   AGE
            granite   0/1     1            0           2s
            granite   1/1     1            1           14s
            ```
5. Create a `Service` CR for the model inference. For example:

    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: granite
      namespace: rhaiis-namespace
    spec:
      selector:
        app: granite
      ports:
        - protocol: TCP
          port: 80
          targetPort: 8000
    ```
6. Optional: Create a `Route` CR to enable public access to the model. For example:

    ```yaml
    apiVersion: route.openshift.io/v1
    kind: Route
    metadata:
      name: granite
      namespace: rhaiis-namespace
    spec:
      to:
        kind: Service
        name: granite
      port:
        targetPort: 80
    ```
7. Get the URL for the exposed route. Run the following command:

    ```terminal
    $ oc get route granite -n rhaiis-namespace -o jsonpath='{.spec.host}'
    ```

    **Example output**

    ```terminal
    granite-rhaiis-namespace.apps.example.com
    ```

Ensure that the deployment is successful by querying the model.
Run the following command:

```terminal
curl -X POST http://granite-rhaiis-namespace.apps.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.1-2b-instruct-quantized.w8a8",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.1
  }'
```

* [Understanding deployments](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/building_applications/deployments#what-deployments-are)
