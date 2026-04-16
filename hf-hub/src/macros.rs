#[allow(unused_macros)]
macro_rules! sync_api {
    (
        $(#[$impl_meta:meta])*
        impl $async_type:ident -> $sync_type:ident {
            $(
                fn $name:ident(&self $(, $pname:ident : $ptype:ty)*) -> $ret:ty;
            )*
        }
    ) => {
        #[cfg(feature = "blocking")]
        $(#[$impl_meta])*
        impl $crate::blocking::$sync_type {
            $(
                #[doc = concat!("Blocking wrapper around [`", stringify!($async_type), "::", stringify!($name), "`].")]
                pub fn $name(&self $(, $pname : $ptype)*) -> $ret {
                    self.runtime.block_on(self.inner.$name($($pname),*))
                }
            )*
        }
    };
}

#[allow(unused_macros)]
macro_rules! sync_api_stream {
    (
        $(#[$impl_meta:meta])*
        impl $async_type:ident -> $sync_type:ident {
            $(
                fn $name:ident(&self $(, $pname:ident : $ptype:ty)*) -> $item:ty;
            )*
        }
    ) => {
        #[cfg(feature = "blocking")]
        $(#[$impl_meta])*
        impl $crate::blocking::$sync_type {
            $(
                #[doc = concat!("Blocking wrapper around [`", stringify!($async_type), "::", stringify!($name), "`]. Collects all items into a `Vec`.")]
                pub fn $name(&self $(, $pname : $ptype)*) -> $crate::error::Result<Vec<$item>> {
                    use futures::StreamExt;
                    self.runtime.block_on(async {
                        let stream = self.inner.$name($($pname),*)?;
                        futures::pin_mut!(stream);
                        let mut items = Vec::new();
                        while let Some(item) = stream.next().await {
                            items.push(item?);
                        }
                        Ok(items)
                    })
                }
            )*
        }
    };
}

#[allow(unused_macros)]
macro_rules! sync_api_async_stream {
    (
        $(#[$impl_meta:meta])*
        impl $async_type:ident -> $sync_type:ident {
            $(
                fn $name:ident(&self $(, $pname:ident : $ptype:ty)*) -> $item:ty;
            )*
        }
    ) => {
        #[cfg(feature = "blocking")]
        $(#[$impl_meta])*
        impl $crate::blocking::$sync_type {
            $(
                #[doc = concat!("Blocking wrapper around [`", stringify!($async_type), "::", stringify!($name), "`]. Collects all items into a `Vec`.")]
                pub fn $name(&self $(, $pname : $ptype)*) -> $crate::error::Result<Vec<$item>> {
                    use futures::StreamExt;
                    self.runtime.block_on(async {
                        let stream = self.inner.$name($($pname),*).await?;
                        futures::pin_mut!(stream);
                        let mut items = Vec::new();
                        while let Some(item) = stream.next().await {
                            items.push(item?);
                        }
                        Ok(items)
                    })
                }
            )*
        }
    };
}
