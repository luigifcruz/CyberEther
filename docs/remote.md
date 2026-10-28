---
title: Remote Streaming
description: Stream the CyberEther interface to a browser on another device, hosted entirely on your own machine.
order: 42
category: Usage
parent: platforms
---

CyberEther Remote streams the full interface of a running instance to a web browser on another device and sends keyboard and mouse input back. The flowgraph keeps running on the host, with every native backend and device that machine has, while the viewer only needs a browser. It is the way to drive a headless server, a radio on a rooftop, or a GPU box in another room from a laptop or a phone.

Everything is self-hosted. When streaming starts, CyberEther launches a small broker on the host that serves the viewer page and introduces viewers to the host. Once a viewer is approved, video and input flow directly between the browser and the host over WebRTC. No account, cloud service, or third party is involved.

## Quick Start

Remote Streaming needs the Vulkan renderer on the host. Linux and Windows use it by default. On macOS, where Metal is the default, start CyberEther with `--renderer vulkan`.

### From the Desktop App

1. Open the Start tab. It appears whenever no flowgraph is open, and `CTRL+T` or the **+** button next to the tabs opens it at any time.
2. In the **Remote** section of the sidebar, click **Start Streaming**.
3. Click **Copy Link** and open the link on the viewing device, or click **QR Code** and scan it with a phone camera.
4. While it waits, the viewer shows a six character code. The same code appears in the viewer list of the **Remote** section. Check that they match, then click the check mark next to it.

While streaming, the **Remote Sharing** button in the titlebar brings back the Start tab. Click **Stop Streaming** to end the session and disconnect every viewer, or click the X next to a viewer to disconnect only that one. Builds without remote streaming hide the **Remote** section.

### From the Command Line

For a headless machine, start CyberEther with the remote option and a flowgraph:

```
cyberether --remote --headless flowgraph.yml
```

On macOS, add `--renderer vulkan` to the command.

The terminal prints a QR code, the invite link, and the access token. When a viewer connects, the terminal shows its code and asks whether to approve it. Answer `y` after checking that the code matches the one on the viewer.

## How It Works

The broker is part of CyberEther itself and listens on port 8686 by default. It does three things:

- It serves the viewer page at `/remote`. The page is downloaded from the CyberEther CDN when CyberEther is built and is embedded in the binary, so nothing is fetched at runtime.
- It hosts the signalling channel at `/api/v1/remote/signaller`, which the viewer and the host use to exchange connection details.
- It holds the waiting list of viewers and enforces approval.

The invite link has the form `http://<host>:8686/remote#<token>`. The token after the `#` stays in the browser and is never sent in the page request. Anyone holding a valid link can ask to join, and nobody gets in until the host approves them, unless auto approval is turned on.

After approval the broker steps aside. Video and input travel over a direct WebRTC connection between the viewer and the host, encrypted end to end.

## Network Requirements

The viewer must be able to reach the host directly. There is no relay, so both devices need to be on the same network, or on networks that can route to each other.

- **Same local network** works out of the box.
- **Different networks** work over a VPN such as WireGuard or Tailscale, which puts both devices on one routable network.
- **Locked down hosts** that only accept SSH can stream through an SSH tunnel, as described in [Streaming Over SSH](#streaming-over-ssh).
- **Firewalls** on the host must allow the broker port (TCP 8686 by default) and the media connection. Media uses UDP, and also TCP when it is allowed. Set a **Media Port Range** to open a fixed set of ports instead of random ones.

> The broker is designed for trusted local networks. Do not expose it directly to the internet, for example through port forwarding. Use a VPN to reach the host from outside.

## Settings

Open **Settings** and select **Remote**, or click the settings button in the **Remote** section of the Start tab. Changes apply the next time streaming starts. Every setting has a matching command line option, which overrides the saved value for that run only and leaves the saved settings untouched.

### Access

| Setting | Option | Default | Meaning |
|---|---|---|---|
| **Viewer Approval** | `--remote-auto-approve <on\|off>` | Manual | Whether new viewers join automatically or wait for approval. |

Leave approval on manual unless every device that could see the invite link is trusted.

### Video

| Setting | Option | Default | Meaning |
|---|---|---|---|
| **Video Framerate** | `--remote-framerate <fps>` | 30 | Target framerate of the stream. |
| **Video Codec** | `--remote-codec <codec>` | H264 | Compression format. Choices are h264, vp8, and vp9. The av1 choice is not supported for streaming yet. |
| **Video Encoder** | `--remote-encoder <encoder>` | Auto | Encoder used for the stream. Auto picks a hardware encoder when one is available. |

Some phone browsers cannot decode the H264 profile produced by hardware encoders. When a phone connects but never shows video, switch the codec to VP8, which every browser supports.

### Broker

| Setting | Option | Default | Meaning |
|---|---|---|---|
| **Broker Address** | `--remote-broker-address <ip>` | 0.0.0.0 | Address the broker listens on. The default accepts viewers on every network interface. |
| **Broker Port** | `--remote-broker-port <port>` | 8686 | Port that serves the viewer page and the signalling channel. |
| **Broker Public URL** | `--remote-broker-public-url <url>` | Automatic | Address placed in the invite link. |
| **Broker TLS Certificate** | `--remote-broker-tls-cert <path>` | None | Certificate file in PEM format. Enables HTTPS. |
| **Broker TLS Key** | `--remote-broker-tls-key <path>` | None | Private key in PEM format that matches the certificate. |

To accept viewers from a single network only, set **Broker Address** to the address of that interface. Setting it to 127.0.0.1 limits viewers to the host itself.

When **Broker Public URL** is empty, CyberEther builds the invite link from the local network address of the host and the broker port. Set it when viewers reach the host by a different name, for example a hostname that matches a certificate or the address of a VPN interface. It takes an origin only, written as scheme, host, and port, such as `https://cyberether.lan:8686`. Paths are not supported, since the viewer always connects at the root of the origin.

### Media

| Setting | Option | Default | Meaning |
|---|---|---|---|
| **Media Transport** | `--remote-media-transport <transport>` | Auto | Network protocols offered for the video connection. Choices are auto, udp, and tcp. |
| **Media Port Range** | `--remote-media-port-range <min-max>` | Any | Local ports used for the video connection, such as 40000-40100. |
| **Media Address** | `--remote-media-address <ip>` | Automatic | The only local IP address offered for the video connection. Leave empty to offer every address. |

Auto offers both UDP and TCP and lets the connection pick the best one, which is usually UDP. Choose TCP on networks that block UDP. Expect more latency and stutter on TCP, since lost packets hold up the frames behind them.

Give the port range more than one port. Each viewer needs its own port, and a port that was just released can take a moment before it is usable again.

By default the host offers every network address it has, and the viewer picks one it can reach. **Media Address** narrows that to a single address. It is mainly useful for SSH tunnels, where it is set to 127.0.0.1.

## Securing the Connection

By default the broker serves plain HTTP. The video connection is always encrypted, but the viewer page and the signalling channel are not. Anyone who can watch traffic on the network can read the invite token or tamper with the viewer page. On a home network that is often acceptable. On a shared network, enable TLS.

To enable TLS, set both **Broker TLS Certificate** and **Broker TLS Key**. The broker then serves HTTPS and secure WebSockets on the same port. Set **Broker Public URL** to an `https://` address whose hostname matches the certificate, otherwise browsers reject the page.

A few ways to get a certificate:

- **A local certificate authority.** Tools such as mkcert create a certificate for a hostname or address on your network and install a matching authority on your devices, so browsers trust it without warnings.
- **A certificate for a domain you own.** When the host has a DNS name, a certificate from a public authority works on every device without installing anything.
- **A reverse proxy.** A proxy such as Caddy or nginx can terminate TLS in front of the broker. Forward the whole origin to the broker, including WebSocket upgrades, and set **Broker Public URL** to the address of the proxy. Serving the broker under a path prefix is not supported.

Keep the private key readable only by the account that runs CyberEther.

## Streaming Over SSH

When the host sits behind a firewall that only allows SSH, the whole session can travel through one SSH connection. The broker and the video connection both run over TCP, so SSH can forward them to the viewing machine, and the browser connects to them as if the host were local.

The video connection needs three things to fit through the tunnel. It must use TCP, because SSH does not forward UDP. It must use a small, known set of ports, so each one can be forwarded. And the host must offer only its loopback address, so the browser connects to the forwarded port on the viewing machine instead of trying to reach the host directly.

### On the Host

Start CyberEther with the broker and the video connection limited to the loopback address:

```
cyberether --remote --headless \
  --remote-broker-address 127.0.0.1 \
  --remote-broker-public-url http://localhost:8686 \
  --remote-media-transport tcp \
  --remote-media-port-range 40000-40003 \
  --remote-media-address 127.0.0.1 \
  flowgraph.yml
```

The same values can be saved in **Settings** under **Remote** instead. Each port in the range carries one viewer, so four ports allow up to four viewers at a time. Keep one spare, since a port that was just released can take a moment to become available again.

Run this command from inside an SSH session to the host, so the approval prompt appears in your terminal. Alternatively, add `--remote-auto-approve on`, since only someone with access to the tunnel can reach the broker.

### On the Viewing Machine

Open a second SSH connection that forwards the broker port and every media port to the same port numbers on the host:

```
ssh -N \
  -L 8686:127.0.0.1:8686 \
  -L 40000:127.0.0.1:40000 \
  -L 40001:127.0.0.1:40001 \
  -L 40002:127.0.0.1:40002 \
  -L 40003:127.0.0.1:40003 \
  user@host
```

The local and remote port numbers must match. The host advertises its own port numbers to the browser, and the browser connects to those same numbers on the viewing machine. Make sure nothing else on the viewing machine already uses these ports.

With the tunnel open, open the invite link printed by the host, such as `http://localhost:8686/remote#<token>`, in a browser on the viewing machine.

### Notes

- Expect more latency than a direct connection. Video over TCP pauses briefly whenever a packet has to be resent, and SSH adds its own overhead.
- Lowering **Video Framerate** helps on slow links.
- Because the broker listens only on the loopback address, nothing is exposed on the network of the host. All traffic, including the viewer page and the invite token, is encrypted by SSH.

## Troubleshooting

**The invite link does not open on the other device.** Check that both devices are on the same network, that the host firewall allows the broker port, and that the address in the link belongs to the host. When the host has several network interfaces, set **Broker Public URL** to the address the viewer can reach.

**The viewer waits forever.** The host has not approved it yet. Look for the matching code in the **Remote** section of the Start tab or in the terminal.

**The viewer connects but shows no video.** On phones this is usually the codec. Switch **Video Codec** to VP8. On restrictive networks, try **Media Transport** set to TCP, and check that the firewall allows the media ports.

**Streaming fails to start.** On macOS, check that CyberEther runs with `--renderer vulkan`. Otherwise, another program may already be using the broker port. Pick a different **Broker Port**. When TLS is enabled, check that both files exist and that the key matches the certificate.
